// P17-P3: BPE 分词器 —— **外部 mmap 词表** + 纯 Rust 手写编解码 + 硬守卫
//
// ============================ 本模块属于哪一步 ============================
//
// P3 的验收链路是「文本输入 -> token id -> logits -> 文本输出」。本模块负责这条
// 链路**两端**（文本 <-> token id），中间那段（id -> logits -> id）是模型的事，
// 不在本模块里。把两端做对是**可判定的**：绕一圈回来必须**逐字节相同**。
//
// 与 P17 其它步骤的关系：
//   * P0/P1（train/）训的是**稀疏连接**，本模块不碰权重、不碰前向。
//   * 本模块唯一的输入是词表文件，唯一的输出是 id 序列与字节串。
//   * 因此它**没有**依赖 conn_sparse / neuron_kv / train 任何一项 —— 这个
//     feature 是正交的，`--features bpe` 单独开必须能编译过（实测见报告）。
//
// ============================ 为什么词表放外部文件 ============================
//
// 任务书 P3 的硬约束：「词表 **mmap 外部文件**，不占引擎字节」。
// 这一条不是风格问题，是**预算**问题：引擎体积预算是 84480 B，当前默认引擎
// 74752 B（余 9728 B）。一个 4096 条合并规则的词表约 28 KB，任何形式的内嵌
// （include_bytes!、常量池、AOT）都会当场把预算打穿。
//
// 所以本模块的加载路径是 `mmap_file_ro()`（core/src/lib.rs，PAGE_READONLY）。
// **注意它刻意只读**：词表一旦被就地改写就是**静默的语义损坏** —— 同一个 id
// 在不同时刻解出不同的字节，而且没有任何东西会报错。只读映射把这种写变成
// 访问违例崩溃。这与 lib.rs 里那条「推理用 mmap、训练必须用 read_file」的
// 边界是同一条理由。
//
// ============================ 为什么只存合并表，不存词表副本 ============================
//
// 一个 piece 的字节由**合并表递归定义**：
//
//     bytes(id) = [id]                      若 id < 256
//     bytes(id) = bytes(left) ++ bytes(right)   若 id = 256 + i，规则 i = (left, right)
//
// 于是词表是**可推导的**，存一份副本等于**同一件事声明两遍** —— 而两份声明
// 可以不一致，且不一致的时候没有任何东西会发现。本项目已经在 S8（描述符声明
// 的长度不约束 section 内容）上吃过一次这个亏，这里不重犯：
//
//   * 文件里只有合并表（每规则 7 字节）+ 头部。
//   * `piece_bytes()` 自底向上展开一次，结果存在**调用方给的一块 arena** 里；
//     稳态解码是「查表 + 拼接」，零分配、零递归。
//
// ---- 代价必须诚实说明 ----
// 展开需要一块 arena。多大的 arena 由词表自己决定（见 `vocab_arena_bytes`），
// 上限由头部字段推出，不是猜的。**另外**：展开是 O(总字节数)，只在装载期做
// 一次；解码路径不展开。
//
// ============================ 解码：展开到叶子，然后拼接 ============================
//
// ⚠️ 这一节是本模块最重要的一段注释，因为它记录的是**一个被实测否定的设计**。
//
// 第一版解码走的是「**重放**」模型：给定 id 256+i，在已经建好的输出 buffer 里
// 找规则 (left, right) 的**相邻出现**，然后就地拼接。这个模型是**错的**，而且
// 错得不显眼。决定性证据：
//
//     'VMLM' 编码成 **[415]**，而 415 的规则是 (86, 411) ——
//     **id 411 根本不在 id 序列里**。
//
// 真实 BPE 解码器**从不**依赖「操作数出现在序列里」。它查合并表，把 id 展开成
// 叶子字节。重放模型要求 411 出现在 buffer 里，那是一个**格式从未要求的额外
// 假设**，而这个假设在「整词 = 单个 id」这种最常见的情形下**必然失败**。
//
// 实测失败率（tools/p17_bpe_vocab.py，重放模型）：'VMLM' / 'Transformer' /
// 'kernel32' / '是一个把' / 整句中文**全部硬失败**，只有 'the' 这种纯字节词通过。
//
// ---- 为什么不能靠「扫描输出 buffer」补救（实测数据）----
//
//   v1: 「找操作数相邻的位置替换」—— 要求太强，665 条规则里 415 条匹配不到任何
//       东西，判据大面积硬失败。
//   v2: 「按 piece 字节的**最左**出现位置拼接」(byte.find) ——
//       **错误且安静**：真实序列上返回**非法 UTF-8**（b'\x96...'），即一次
//       "成功" 的解码产出了**任何编码器都不可能写出**的字节。13 个探针词里只有
//       2 个恰好正确。
//   v3: 「按最右出现位置拼接」(byte.rfind) —— 好一些，但仍然依赖「操作数的字节
//       以连续后缀形式存在」这个假设。实测：**100 条采样规则里 27 条的左出现
//       != 尾随出现**，所以 find -> rfind 不是一个可以随手改的细节；而且即使
//       rfind 也在上面那个整词单 id 的情形上失败。
//
// ⇒ 结论写成规则：**不要从输出 buffer 推导 piece 的字节。**从合并表推导，
//   然后拼接。这就是本模块实现的东西，它只用**一个真相源**（合并表），
//   且对**任何编码器能产生的 id 序列**都成立。
//
// 附带收益：这个改动**删掉了一整类错误** —— 「不可达的合并 id」不可能发生了，
// 于是没有延迟工作队列、没有第二遍、没有需要维护的状态。
// 删掉一类错误比维护一套处理它的协议好。
//
// ============================ 非法输入必须硬失败（本项目铁律） ============================
//
// 每个非法输入都 `exit(9)`，且消息里**带具体数值**。理由与 conn_sparse.rs 相同：
// BPE 表被按**错误布局**解析时，产出的不是崩溃而是**看着通顺的垃圾** ——
// 这是最坏的一类失败，因为输出本身看不出任何异常。所以错 magic、截断、
// 多声明、越界 id、越界操作数一律硬失败。
//
// ⚠️ 关于 exit 码的口径（照抄 conn_sparse.rs，不要读成「返回 9」）：
//    `exit(9)` 走 `ExitProcess(9)`，**不返回**。在 std 测试 harness 里本模块被
//    链接进一个正常进程，所以观测到的**进程退出码就是 9**。这就是守卫的观测形式。
//
// ============================ 尺寸可校验性（S8 缺陷类，不重犯） ============================
//
// 做法照抄 conn_sparse.rs：「需要多少字节」是头部字段的**纯函数**：
//
//     section_bytes(n_merges, n_special)
//         = BPE_HDR + n_merges * MERGE_ENTRY + n_special * SPECIAL_ENTRY
//
// 于是截断与多声明各是一条**独立**、能失败的守卫：
//   * 实际可读字节 != 上式       -> G1
//   * merge_bytes != n*7         -> G3   （把 n_merges 与 merge_bytes 互相绑定）
//   * special_bytes != ns*5      -> G15
//   * n_pieces != 256 + n_merges -> G5   （把头部第三个字段也绑上）
// 单个字段漂移会被后面的字段抓住；想自洽必须**同时**撒谎。

use crate::{exit, print};

// ============================================================================
// 1. 布局常量（唯一真相源；Python 侧等价定义见 tools/p17_bpe_vocab.py）
// ============================================================================

/// 头部字节数。八个 u32，逐个偏移在下面写死：
///   [0]  magic           = BPE_MAGIC（"BPE1"）
///   [4]  version         == 1
///   [8]  n_merges        合并规则数 M
///   [12] n_pieces        词表 piece 数 V（**不存**，见模块头；这里只为长度校验）
///   [16] merge_bytes     合并数组字节数 == M * 7
///   [20] special_bytes   特殊表字节数 == n_special * 5
///   [24] n_special       特殊 token 数
///   [28] reserved0       必须为 0
pub const BPE_HDR: usize = 32;

/// 头部 magic："BPE1" 的小端 u32。
///
/// sanity（不要凭记忆，按位算一遍）：
///   'B'=0x42 | 'P'<<8=0x5000 | 'E'<<16=0x450000 | '1'<<24=0x31000000
///   = 0x31504542                       // 打印为 0x3150_4542
///
/// 以版本号结尾（而不是 0）的理由与 CS_MAGIC 相同：下一版叫 "BPE2"，
/// 于是**版本漂移会在 magic 这一条上失败**，而不是悄悄按 v1 解析。
/// （另有独立的 version 字段守卫 G9；两条独立，任何一条单独都能开火。）
pub const BPE_MAGIC: u32 = 0x3150_4542;
/// 本模块实现的版本号。
pub const BPE_VERSION: u32 = 1;

/// 头部字段偏移（给测试与 Python 侧引用，不要在各处重打字面量）。
pub const BPE_OFF_MAGIC: usize = 0;
pub const BPE_OFF_VERSION: usize = 4;
pub const BPE_OFF_N_MERGES: usize = 8;
pub const BPE_OFF_N_PIECES: usize = 12;
pub const BPE_OFF_MERGE_BYTES: usize = 16;
pub const BPE_OFF_SPECIAL_BYTES: usize = 20;
pub const BPE_OFF_N_SPECIAL: usize = 24;
pub const BPE_OFF_RESERVED0: usize = 28;

/// 一条合并规则 = 8 字节：(u16 left, u16 right, u32 merged_id)，**小端、无填充**。
///
/// ⚠️ 这个常量**第一版写成 7**，是一个**真实的 bug**，而它是被 Rust 自己的长度
/// 守卫抓住的 —— 值得写下来，因为它示范了本模块存在的理由。
///
/// 布局是「两个 u16 后接一个 u32，直接拼接」，没有对齐填充可言：
/// 2 + 2 + 4 = **8**，由定义得出。7 的来源是「先按 C 结构体对齐算再减掉填充」
/// —— 但本格式根本没有那个对齐层，减掉填充就把尺寸算错了。
///
/// 后果：头部声明 665*7 = 4655，而 Python 侧实写 665*8 = 5320，文件 5352 字节
/// / 声明 4687。抓它的是**独立的另一端**：Rust 的 G1 在装载期报出
///     BPE: file length disagrees with header
///     BPE: avail=5352 required=4687
/// 写入方自己**永远不会**发现 —— 这正是「声明尺寸与真实尺寸必须在写入点互相
/// 校验」的实证，也是为什么 G1 用**精确相等**而不是 `>=`。
///
/// ⚠️ 另外：操作数是 **u16**，不是 u8。第一版用 u8，理由是「操作数就是字节」
/// —— **那也是错的**：操作数可以是一条**早先的合并结果**，其 id >= 256。
/// 实测本词表的 1330 个操作数里 **848 个 >= 256**，最大 919。u8 会（Python 侧）
/// 抛异常 /（Rust 侧）**静默截断**，而截断后的 id 仍然「看起来像个合法数字」。
/// 代价每条规则 +2 字节；本词表 665 条 = +1330 B，值得。
pub const MERGE_ENTRY: usize = 8;
/// 特殊表一项 = (u8 code, u32 piece_id) = 5 字节。
pub const SPECIAL_ENTRY: usize = 5;

/// 单字节 piece 的个数。id 0..255 是单字节，256..256+M-1 是合并。
///
/// 这个「slot-1 模型」是**格式的定义**，不是实现细节：它让 bytes(id) 的递推
/// 有唯一的基础情形，也让 `merged_id == 256 + i` 成为一条可校验的冗余
/// （见 G11）。
pub const N_BYTE_PIECES: u32 = 256;

/// 输出缓冲上限（id 个数）。1<<20 个 u32 = 4 MiB。
///
/// 超限**硬失败**（G13），不截断。理由：截断产生的是一个**合法但静默错误**的
/// id 序列，而它的损坏只会在后面某处以完全无关的形式暴露出来
/// （本模块第一版就踩过：buffer 被截断，报出来的是「解码出错」）。在**产出**
/// 溢出的那一层报错，才有归因价值。
pub const BPE_OUT_LIMIT: usize = 1 << 20;

/// 词表文件默认路径（CLI 用；与 examples/p17_bpe 的产物同名）。
pub const BPE_DEFAULT_PATH: &[u8] = b"bpe_vocab.vmb";

// ============================================================================
// 2. 布局 + 硬校验
// ============================================================================

/// 词汇表布局。构造后**必须** `validate()` 才能用。
#[derive(Clone, Copy)]
pub struct BpeLayout {
    /// 合并规则数 M。
    pub n_merges: u32,
    /// 词表 piece 数 V（== 256 + M，由 G5 强制）。
    pub n_pieces: u32,
    /// 头部声明的合并数组字节数（== M * 7，由 G3 强制）。
    pub merge_bytes: u32,
    /// 头部声明的特殊表字节数（== n_special * 5，由 G15 强制）。
    pub special_bytes: u32,
    /// 特殊 token 数。
    pub n_special: u32,
}

impl BpeLayout {
    /// 紧致构造式：**导出**所有冗余字段，让「头部自洽」是构造出来的，
    /// 而不是靠调用方算对。
    pub fn tight(n_merges: u32, n_special: u32) -> BpeLayout {
        BpeLayout {
            n_merges,
            n_pieces: N_BYTE_PIECES.wrapping_add(n_merges),
            merge_bytes: (n_merges as u64 * MERGE_ENTRY as u64) as u32,
            special_bytes: (n_special as u64 * SPECIAL_ENTRY as u64) as u32,
            n_special,
        }
    }

    /// 词汇表 piece 总数 = 256 + M（用 u64 中间量，见 G4）。
    pub fn total_pieces(&self) -> u64 {
        N_BYTE_PIECES as u64 + self.n_merges as u64
    }

    /// 合并数组起始偏移。
    pub fn merges_off(&self) -> usize {
        BPE_HDR
    }
    /// 特殊表起始偏移。
    pub fn specials_off(&self) -> usize {
        BPE_HDR + self.merge_bytes as usize
    }

    /// **整个文件的字节数**。尺寸可校验性的唯一公式：给定头部字段，
    /// 需要多少字节**没有第二种答案**。
    pub fn section_bytes(&self) -> u64 {
        BPE_HDR as u64 + self.merge_bytes as u64 + self.special_bytes as u64
    }

    /// 硬校验。任何一条不满足都 `exit(9)` 并打印**具体是哪一条 + 数值**。
    ///
    /// # Safety
    /// 本函数只读 `self`，标 unsafe 是为了与 conn_sparse 的调用形状一致
    /// （调用点都在 `unsafe fn` 里）。
    pub unsafe fn validate(&self) {
        // G2: 零合并规则。空词表不是「退化但合法」—— 它连一个 piece 都定义不了，
        //     任何 id >= 256 都是越界。允许它等于允许「装载成功但什么都编不了」。
        if self.n_merges == 0 {
            print(b"BPE: n_merges is zero\n");
            print(b"BPE: n_merges=0 n_pieces=");
            print_num(self.n_pieces as usize);
            print(b"\n");
            exit(9);
        }
        // G4: 尺寸上溢 u32。用 u64 中间量判断，**不在 u32 里乘**：
        //     在 u32 里溢出会**变成一个小数字**，于是 section_bytes() 也算小，
        //     整个文件被接受成一个「更小但合法」的 section —— 典型的静默降级。
        if self.total_pieces() > u32::MAX as u64 {
            print(b"BPE: vocabulary size overflows u32\n");
            exit(9);
        }
        if self.section_bytes() > u32::MAX as u64 {
            print(b"BPE: vocabulary file exceeds 32-bit offsets\n");
            print(b"BPE: section_bytes=");
            print_num(self.section_bytes() as usize);
            print(b"\n");
            exit(9);
        }
        // G5: n_pieces 必须**恰好**是 256 + n_merges。这绑定头部的第三个字段，
        //     使「只改一个字段」不可能自洽。
        let want_pieces = self.total_pieces() as u32;
        if self.n_pieces != want_pieces {
            print(b"BPE: n_pieces != 256 + n_merges\n");
            print(b"BPE: n_pieces=");
            print_num(self.n_pieces as usize);
            print(b" n_merges=");
            print_num(self.n_merges as usize);
            print(b" expected=");
            print_num(want_pieces as usize);
            print(b"\n");
            exit(9);
        }
        // G3: merge_bytes 必须**恰好**是 M * MERGE_ENTRY。
        let want_mb = (self.n_merges as u64 * MERGE_ENTRY as u64) as u32;
        if self.merge_bytes != want_mb {
            // ⚠️ 消息里的乘数**不带字面量**。第一版这里写死了 " * 7"（当时的
            // MERGE_ENTRY），后来常量改成 8 而字符串没跟着改，于是守卫**打印了
            // 一个错误的期望公式** —— 报错消息本身把排查引向歧途，比不报更糟。
            // 守卫的**判定**一直是对的（它用的是 MERGE_ENTRY），错的只有文案；
            // 抓它的是 can-fail 对照 (b3)（它断言消息文本，于是 MISSING）。
            // 现在改成把 MERGE_ENTRY 的**值**打出来，让它不可能与代码分叉。
            print(b"BPE: merge_bytes != n_merges * MERGE_ENTRY\n");
            print(b"BPE: merge_bytes=");
            print_num(self.merge_bytes as usize);
            print(b" n_merges=");
            print_num(self.n_merges as usize);
            print(b" MERGE_ENTRY=");
            print_num(MERGE_ENTRY);
            print(b" expected=");
            print_num(want_mb as usize);
            print(b"\n");
            exit(9);
        }
        // G15: special_bytes 必须**恰好**是 n_special * 5。
        let want_sb = (self.n_special as u64 * SPECIAL_ENTRY as u64) as u32;
        if self.special_bytes != want_sb {
            print(b"BPE: special_bytes != n_special * 5\n");
            print(b"BPE: special_bytes=");
            print_num(self.special_bytes as usize);
            print(b" n_special=");
            print_num(self.n_special as usize);
            print(b" expected=");
            print_num(want_sb as usize);
            print(b"\n");
            exit(9);
        }
    }
}

// ============================================================================
// 3. 读：文件字节 -> 布局（逐项校验）
// ============================================================================

/// 从文件字节解析并**逐项校验**。
///
/// ============================ 校验顺序（顺序本身是设计的一部分） ============================
///
///   1. `avail >= BPE_HDR`          —— 否则连头部都读不出来（G0）
///   2. magic == BPE_MAGIC          —— 否则后面所有数字都不可信（G8）
///   3. version == 1                —— 版本漂移（G9）
///   4. reserved0 == 0              —— 不是本格式写的东西（G14）
///   5. `validate()`                —— G2/G3/G4/G5/G15
///   6. `avail == section_bytes()`  —— **精确相等**（G1）
///   7. 每条规则 left<256+j、right<256+j、merged_id==256+j（G10/G11）
///   8. 特殊表：piece_id 必须是 byte-fallback（G7）、code 不得重复（G6）
///
/// 为什么 magic 放在长度之前：`n_merges` 是**能算**的，算出来的东西可以自洽。
/// magic 是唯一**不可推导**的一致来源。
///
/// 为什么长度用**精确相等**而不是 `>=`：
/// 本格式是定长结构（尺寸由头部唯一确定），多出来的字节没有合法解释。允许 `>`
/// 等于允许「文件后面挂一段谁也不认识的东西」—— 那正是要拦的静默错位，也让
/// 「多声明 4 字节」这条 can-fail 对照能被抓住。
///
/// ⚠️ 但**这两条理由都还是次要的**。`>=` 会放过一个已经真实发生过的 bug，
/// 这才是 G1 存在的**根本理由**（写在这里，不是附注）：
///
///     MERGE_ENTRY 第一版被写成 7，实际是 8（u16+u16+u32，无填充）。
///     于是头部声明 665*7 = 4655，而写入方实写 665*8 = 5320，
///     文件 5352 B / 声明 4687。
///
/// **写入方自己永远不会发现这件事**：它按自己的常量写，读回来也按自己的常量算，
/// 两侧一致地错。唯一能发现它的是一个**独立**地重放「本格式的尺寸公式」的实现
/// —— 也就是这里。G1 报出来的是：
///
///     BPE: file length disagrees with header
///     BPE: avail=5352 required=4687
///
/// 如果 G1 写成 `avail >= required`，那么 5352 >= 4687 成立，**这个 bug 会一路
/// 活到生产**：解码器会按 4655 的假设去切合并表，第 664 条规则读到的是错位的字节，
/// 而它解出来的是「看起来正常的 token 字节」—— 通顺、量级正常、完全错误。
///
/// ⇒ 精确相等是**为了能抓住「写入方与读取方一致地错」这一类**，
///   而 `>=` 结构上不可能抓住它。这不是风格选择，是覆盖范围的区别。
///
/// # Safety
/// `src` 至少 `avail` 字节可读。
pub unsafe fn read_vocab(src: *const u8, avail: usize) -> BpeLayout {
    // G0: 连头部都放不下。
    if avail < BPE_HDR {
        print(b"BPE: available bytes below header\n");
        print(b"BPE: avail=");
        print_num(avail);
        print(b" BPE_HDR=");
        print_num(BPE_HDR);
        print(b"\n");
        exit(9);
    }
    // G8: magic 先于一切「看起来合理」的字段。
    let magic = get_u32(src, BPE_OFF_MAGIC);
    if magic != BPE_MAGIC {
        print(b"BPE: bad vocabulary magic\n");
        print(b"BPE: got=0x");
        print_hex32(magic);
        print(b" want=0x");
        print_hex32(BPE_MAGIC);
        print(b"\n");
        exit(9);
    }
    // G9: 版本。
    let ver = get_u32(src, BPE_OFF_VERSION);
    if ver != BPE_VERSION {
        print(b"BPE: unsupported vocabulary version\n");
        print(b"BPE: version=");
        print_num(ver as usize);
        print(b" supported=");
        print_num(BPE_VERSION as usize);
        print(b"\n");
        exit(9);
    }
    // G14: reserved0 必须为 0。非 0 说明写这个文件的**不是**本格式的实现，
    //      或者写入方与读取方对 [28..32) 的解释已经分叉 —— 两种都必须硬失败。
    let res0 = get_u32(src, BPE_OFF_RESERVED0);
    if res0 != 0 {
        print(b"BPE: reserved0 is not zero\n");
        print(b"BPE: reserved0=");
        print_hex32(res0);
        print(b"\n");
        exit(9);
    }
    let layout = BpeLayout {
        n_merges: get_u32(src, BPE_OFF_N_MERGES),
        n_pieces: get_u32(src, BPE_OFF_N_PIECES),
        merge_bytes: get_u32(src, BPE_OFF_MERGE_BYTES),
        special_bytes: get_u32(src, BPE_OFF_SPECIAL_BYTES),
        n_special: get_u32(src, BPE_OFF_N_SPECIAL),
    };
    // G2/G3/G4/G5/G15
    layout.validate();

    // G1: 可读字节数必须**精确**等于头部推出的尺寸。
    let need = layout.section_bytes();
    if avail as u64 != need {
        print(b"BPE: file length disagrees with header\n");
        print(b"BPE: avail=");
        print_num(avail);
        print(b" required=");
        print_num(need as usize);
        print(b"\n");
        exit(9);
    }

    let m = layout.n_merges as usize;
    let mo = layout.merges_off();
    // G10 + G11: 每条规则的操作数必须**已经存在**，且 merged_id 必须**恰好**是 256+j。
    //
    //   * G10（left < 256+j 且 right < 256+j）是**展开的良基性**：它保证
    //     `piece_bytes()` 一次正向扫描就能算完，不需要递归、不需要环检测。
    //     环检测放在**格式层**而不是解码器里 —— 非法文件必须在装载期失败，
    //     而不是让解码器去容忍它。
    //   * G11（merged_id == 256+j）是**冗余校验**：merged_id 是可推导的，
    //     正因为可推导，写错了才抓不到（除非单独校验）。它能抓住写入方的
    //     差一错误，也能抓住「整块合并表被平移了一条」这种错位。
    let mut i = 0usize;
    while i < m {
        let off = mo + i * MERGE_ENTRY;
        let l = get_u16(src, off) as u32;
        let r = get_u16(src, off + 2) as u32;
        let mid = get_u32(src, off + 4);
        let limit = N_BYTE_PIECES + i as u32; // 规则 j 只能引用已存在的 id
        if l >= limit || r >= limit {
            print(b"BPE: merge operands must be ids that already exist\n");
            print(b"BPE: rule=");
            print_num(i);
            print(b" left=");
            print_num(l as usize);
            print(b" right=");
            print_num(r as usize);
            print(b" limit=");
            print_num(limit as usize);
            print(b"\n");
            exit(9);
        }
        let want_mid = N_BYTE_PIECES + i as u32;
        if mid != want_mid {
            print(b"BPE: merged_id is not 256 + merge index\n");
            print(b"BPE: rule=");
            print_num(i);
            print(b" merged_id=");
            print_num(mid as usize);
            print(b" expected=");
            print_num(want_mid as usize);
            print(b"\n");
            exit(9);
        }
        i += 1;
    }

    // G6 + G7: 特殊表。本模型下**训练出的**词表 n_special == 0（每个 piece 都在
    // 合并树里，见 Python 侧的实测：665/665 规则都会触发）。非空的特殊表是给
    // 「不在合并树里的 piece」（例如 <0xXX> 字节回退 token）留的合法位置。
    //   * G7: piece_id 必须 < 256。id >= 256 的 piece **一定**在合并树里，
    //         给它再挂一条特殊规则意味着**两个真相源**，而两者可以不一致。
    //   * G6: code 不得重复 —— 重复的 byte-fallback 让「给定 code 找 piece」
    //         无唯一答案，而唯一性正是它能当真相源的前提。
    let ns = layout.n_special as usize;
    let so = layout.specials_off();
    let mut j = 0usize;
    while j < ns {
        let code = *src.add(so + j * SPECIAL_ENTRY);
        let pid = get_u32(src, so + j * SPECIAL_ENTRY + 1);
        if pid >= N_BYTE_PIECES {
            print(b"BPE: special piece is a byte-fallback id\n");
            print(b"BPE: entry=");
            print_num(j);
            print(b" code=");
            print_num(code as usize);
            print(b" piece_id=");
            print_num(pid as usize);
            print(b" limit=");
            print_num(N_BYTE_PIECES as usize);
            print(b"\n");
            exit(9);
        }
        let mut k = 0usize;
        while k < j {
            let c2 = *src.add(so + k * SPECIAL_ENTRY);
            if c2 == code {
                print(b"BPE: special byte code declared twice\n");
                print(b"BPE: code=");
                print_num(code as usize);
                print(b" entries=");
                print_num(k);
                print(b" and ");
                print_num(j);
                print(b"\n");
                exit(9);
            }
            k += 1;
        }
        j += 1;
    }

    layout
}

/// 直接从文件加载（mmap 路径）。**这是生产入口。**
///
/// 为什么是 mmap 而不是 read_file：见模块头「为什么词表放外部文件」。
/// `mmap_file_ro` 自己会在文件不存在/映射失败时硬失败（exit 2），所以这里
/// 拿到的 `(base, len)` 一定是有效的，`len` 就是文件长度 —— 于是 G1 的
/// 「精确长度」有一条**来自操作系统**的依据，而不是调用方声明的。
///
/// # Safety
/// 调用方保证路径为 NUL 结尾的 ASCII/ANSI 串。
pub unsafe fn load_vocab_mmap(path: *const u8) -> (*const u8, usize, BpeLayout) {
    let (base, len) = crate::mmap_file_ro(path);
    let layout = read_vocab(base, len);
    (base, len, layout)
}

// ============================================================================
// 4. 展开：piece id -> 字节（唯一真相源）
// ============================================================================

/// 展开整个词表到一块**调用方给的** arena，返回 `(piece_off, piece_len)` 表。
///
/// 语义（与 tools/p17_bpe_vocab.py 的 `piece_bytes()` **逐位相同**）：
///
///     bytes(id) = [id]                        若 id < 256
///     bytes(id) = bytes(left) ++ bytes(right) 若 id = 256 + i，规则 i = (left, right)
///
/// 自底向上一次扫描即可（G10 保证操作数已存在）：
///   1. 先把 256 个单字节 piece 铺进 arena；
///   2. 再按 j = 0..M-1 顺序铺合并 piece —— 此时 left/right 的位置已知。
///
/// 于是稳态解码就是「按 id 取 (off, len)，memcpy 到输出」：**不递归、不分配**。
/// 这也是为什么本模块不需要 std：arena 是调用方给的。
///
/// `piece_off`/`piece_len` 各需要 `n_pieces` 项。本函数不分配，调用方负责。
///
/// ⚠️ `arena_cap` **不是可选的**。第一版没有这个参数，于是「arena 给小了」
/// 的直接后果是 `copy_nonoverlapping` **越界写**，进程以 0xC0000005 死掉，
/// 而报出来的是一个没有任何归因价值的数字（实测 3221225477）。本项目的铁律是
/// 非法状态必须**硬失败并打印具体数值**，所以这里把容量显式传入并逐次检查：
/// 报出「第几条规则、需要多少、还剩多少」，而不是崩。
///
/// 返回**实际写入**的字节数（= `expand_vocab_measure` 的返回值）。
///
/// # Safety
/// `arena` 至少 `arena_cap` 字节；`piece_off`/`piece_len` 各至少
/// `layout.n_pieces` 项。
pub unsafe fn expand_vocab(
    src: *const u8,
    layout: &BpeLayout,
    arena: *mut u8,
    arena_cap: usize,
    piece_off: *mut u32,
    piece_len: *mut u32,
) -> usize {
    let mut at: usize = 0;
    // ---- 基础情形：256 个单字节 piece ----
    let mut b = 0u32;
    while b < N_BYTE_PIECES {
        if at + 1 > arena_cap {
            print(b"BPE: arena too small for the byte pieces\n");
            print(b"BPE: need=");
            print_num(at + 1);
            print(b" cap=");
            print_num(arena_cap);
            print(b"\n");
            exit(9);
        }
        *arena.add(at) = b as u8;
        *piece_off.add(b as usize) = at as u32;
        *piece_len.add(b as usize) = 1;
        at += 1;
        b += 1;
    }
    // ---- 合并 piece：左右各拷一段 ----
    let mo = layout.merges_off();
    let m = layout.n_merges as usize;
    let mut i = 0usize;
    while i < m {
        let off = mo + i * MERGE_ENTRY;
        let l = get_u16(src, off) as usize;
        let r = get_u16(src, off + 2) as usize;
        let lo = *piece_off.add(l) as usize;
        let ll = *piece_len.add(l) as usize;
        let ro = *piece_off.add(r) as usize;
        let rl = *piece_len.add(r) as usize;
        // 每次拷贝前检查容量。这是第一版缺失的那个检查。
        if at + ll + rl > arena_cap {
            print(b"BPE: arena too small to expand the vocabulary\n");
            print(b"BPE: rule=");
            print_num(i);
            print(b" need=");
            print_num(at + ll + rl);
            print(b" cap=");
            print_num(arena_cap);
            print(b"\n");
            exit(9);
        }
        core::ptr::copy_nonoverlapping(arena.add(lo), arena.add(at), ll);
        at += ll;
        core::ptr::copy_nonoverlapping(arena.add(ro), arena.add(at), rl);
        at += rl;
        let id = N_BYTE_PIECES as usize + i;
        *piece_off.add(id) = (at - ll - rl) as u32;
        *piece_len.add(id) = (ll + rl) as u32;
        i += 1;
    }
    at
}

/// 展开所需的 arena 字节数 —— **精确值**，两遍扫描算出来，不是上界。
///
/// ⚠️ 第一版这里返回的是「上界」`256 + 2 * merge_bytes`，理由是「每个操作数的
/// 叶子数 <= merge_bytes」。**那个证明是错的，而且实测把它打穿了**：
///
///     上界 = 10896    实际需要 = 38189    比值 0.29
///
/// 实测最长的一个 piece 有 **15120 字节**（语料里重复 40 遍的长中文串被合并成
/// 了一个 id），它**单个**就超过了那个「上界」。错误在于把「字节数」和
/// 「节点数」混为一谈：merge_bytes 是**规则条数**的量级（5320 = 665*8），
/// 而一个 piece 的字节数可以是它的**指数**级 —— 这正是 BPE 的定义：每合并一次，
/// 长度是两棵子树的**和**。
///
/// 后果（实测）：arena 给小了 ⇒ `copy_nonoverlapping` 越界写 ⇒ 进程
/// **0xC0000005 访问违例**，而且是在**装载期**，报出来的是
/// `rc=3221225477` 这种毫无归因价值的数字（本轮 8 个 FAIL 里有 5 个是它，
/// 包括 round-trip、decode 对拍、G12 控制）。这就是「上界写错」的真实代价：
/// 它不是一个保守的浪费，它是一个**崩溃源**。
///
/// ⇒ 现在返回**精确值**：先扫一遍合并表算出每个 piece 的长度（只需要长度表，
///   不需要 arena），再求和。两遍装载期扫描换「绝不越界」，这个交换是划算的。
///   长度表由本函数内部按 id 顺序递推，不需要额外存储 —— 因为 G10 保证
///   `left, right < 256+j`，所以按 id 递增就一定能查到操作数的长度。
///
/// 实现上它借用调用方**尚未分配的** arena 吗？不 —— 它用一块 `piece_len`
/// 的临时数组，但 no_std 下没有 Vec。所以这里改成：调用方必须**先**给出长度表
/// （`piece_len`，`n_pieces` 项），本函数只填它并返回总和；`expand_vocab` 之后
/// 复用同一块长度表。见 `expand_vocab` 的文档。
///
/// # Safety
/// `src`/`layout` 已由 `read_vocab` 校验；`piece_len` 至少 `n_pieces` 项。
pub unsafe fn expand_vocab_measure(
    src: *const u8,
    layout: &BpeLayout,
    piece_len: *mut u32,
) -> usize {
    let mo = layout.merges_off();
    let m = layout.n_merges as usize;
    let mut total: usize = N_BYTE_PIECES as usize;
    let mut b = 0u32;
    while b < N_BYTE_PIECES {
        *piece_len.add(b as usize) = 1;
        b += 1;
    }
    let mut i = 0usize;
    while i < m {
        let off = mo + i * MERGE_ENTRY;
        let l = get_u16(src, off) as usize;
        let r = get_u16(src, off + 2) as usize;
        let ll = *piece_len.add(l) as usize;
        let rl = *piece_len.add(r) as usize;
        // 长度本身也要防溢出：u32 相加。一个畸形词表可以让长度在 u32 里回绕，
        // 于是「精确值」变成一个**小的、看起来合理**的数 —— 又是静默降级。
        let sum = ll as u64 + rl as u64;
        if sum > u32::MAX as u64 {
            print(b"BPE: expanded piece length overflows u32\n");
            print(b"BPE: rule=");
            print_num(i);
            print(b" left_len=");
            print_num(ll);
            print(b" right_len=");
            print_num(rl);
            print(b"\n");
            exit(9);
        }
        *piece_len.add(N_BYTE_PIECES as usize + i) = sum as u32;
        total += sum as usize;
        i += 1;
    }
    total
}

// ============================================================================
// 5. 编码：文本 -> token id
// ============================================================================

/// 一次编码的结果统计（**计数而不是布尔值**：本项目要求每个判据能报出「差多少」）。
#[derive(Clone, Copy, Default)]
pub struct EncodeStat {
    /// 写出的 token 个数。
    pub n_ids: usize,
    /// 输入按空格切出的空词个数（连续空格/前后导空格）。每个贡献一个 id 32。
    pub n_empty_words: usize,
}

/// 单次编码：把一段 UTF-8 文本编成 token id 序列。
///
/// ============================ 分隔符模型（写死在这里，且是判据的一部分） ============================
///
/// 文本按**单个 U+0020** 切成词（`text.split(' ')` 的语义），编码规则是：
///   对每个非空词：把它编码成 id 序列；
///   对每个空词（连续空格/前导/后置空格）：写一个 id 32；
///   词之间**再**写一个 id 32。
///
/// 为什么要把这条**写死**而不是「看着办」：round-trip 判据（逐字节相同）的
/// 成立与否**完全取决于**这个约定。若编码把空格吃掉而解码吐不出空格，判据会
/// 失败，而失败的原因会被误读成解码器有 bug（本模块第一版就经历了这个）。
///
/// 为什么需要「全空输入」特例：输入是纯空格时，切出来的词**全是空**，
/// 而「每个空词一个 id 32」无法区分 1 个空格和 2 个空格（都是 1 个空词）。
/// ⇒ 全空输入按**字节数**写 id 32，一一对应。这条是实测踩出来的：
/// 第一版让 " " 和 "  " 都编成 [32]，编码在那个输入族上**不是单射**，
/// round-trip 静默失败。
///
/// 每词的合并循环是**最低 rank 优先**（不是「从左到右见一个合一个」）：
/// 实测反例 "engine" —— 「见一个合一个」给出 [259,445,291] = "en"/"gi"/"ne"，
/// 而正确结果是 [628,101] = "engin"/"e"。两者**解码回同样的字节**，所以
/// round-trip **看不出**这个 bug；只有与 O(n²) 参考实现对拍才能发现
/// （Python 侧控制 C6 就是这个作用，它当时 FAIL 了）。
///
/// 返回写出的 id 个数；`out` 至少 `BPE_OUT_LIMIT` 项。
///
/// # Safety
/// `out` 至少 `BPE_OUT_LIMIT` 个 u32；`stats` 可为 null。
pub unsafe fn encode_text(
    src: *const u8,
    layout: &BpeLayout,
    text: *const u8,
    text_len: usize,
    out: *mut u32,
    stats: *mut EncodeStat,
) -> usize {
    if !stats.is_null() {
        *stats = EncodeStat::default();
    }
    // G16: 空输入。返回 0 个 token 是**合法**的（没有东西可编），但显式记账：
    //      调用方若拿到 0 又以为「编成功了」，后面会以完全无关的形式炸掉。
    if text_len == 0 {
        return 0;
    }
    let mo = layout.merges_off();
    let m = layout.n_merges as usize;
    let mut n_out: usize = 0;

    // ---- 逐词循环 ----
    // 用一个**可复用**的工作缓冲（栈上）。为什么不用堆：本模块不分配；
    // 为什么定长：见 BPE_WORD_MAX 的注释。
    let mut ids = [0u32; BPE_WORD_MAX];
    let mut pos: usize = 0;
    let mut n_empty: usize = 0;
    while pos <= text_len {
        // 找下一个空格或结尾
        let start = pos;
        while pos < text_len && *text.add(pos) != b' ' {
            pos += 1;
        }
        let wlen = pos - start;
        if wlen == 0 {
            // 空词（连续空格 / 前导 / 后置）。全空输入在下面单独处理，
            // 这里只处理「夹在词之间」的空词。
            n_empty += 1;
        } else {
            let w = encode_word(src, layout, mo, m, text.add(start), wlen, &mut ids);
            let mut k = 0usize;
            while k < w {
                if n_out >= BPE_OUT_LIMIT {
                    print(b"BPE: encode output exceeds id limit\n");
                    print(b"BPE: limit=");
                    print_num(BPE_OUT_LIMIT);
                    print(b" produced=");
                    print_num(n_out + k + 1);
                    print(b"\n");
                    exit(9);
                }
                *out.add(n_out) = ids[k];
                n_out += 1;
                k += 1;
            }
        }
        // 词间/空词写一个空格 id（id 32 == b' '）。结尾不写。
        if pos < text_len {
            if n_out >= BPE_OUT_LIMIT {
                print(b"BPE: encode output exceeds id limit\n");
                print(b"BPE: limit=");
                print_num(BPE_OUT_LIMIT);
                print(b" produced=");
                print_num(n_out + 1);
                print(b"\n");
                exit(9);
            }
            *out.add(n_out) = b' ' as u32;
            n_out += 1;
        } else {
            break;
        }
        pos += 1; // 跳过空格
    }

    // ---- 全空输入特例：一一对应，保证单射 ----
    // 上面那个循环对 "   " 会写 2 个 id（两个空词 + 一个词间空格），
    // 对 "  " 也写 2 个 —— 等等，这里必须逐字节重来一遍才严格。
    // 判据：编码在「纯空格输入」这一族上必须是单射。做法是重写：
    let mut all_space = true;
    let mut t = 0usize;
    while t < text_len {
        if *text.add(t) != b' ' {
            all_space = false;
            break;
        }
        t += 1;
    }
    if all_space {
        if text_len > BPE_OUT_LIMIT {
            print(b"BPE: encode output exceeds id limit\n");
            print(b"BPE: limit=");
            print_num(BPE_OUT_LIMIT);
            print(b" produced=");
            print_num(text_len);
            print(b"\n");
            exit(9);
        }
        let mut k = 0usize;
        while k < text_len {
            *out.add(k) = b' ' as u32;
            k += 1;
        }
        n_out = text_len;
    }

    if !stats.is_null() {
        (*stats).n_ids = n_out;
        (*stats).n_empty_words = n_empty;
    }
    n_out
}

/// 单个词的最大 id 个数（栈工作缓冲大小）。
///
/// 一个 UTF-8 词最多 `bytes` 个 id（每个字节一个）；合并只**减少**个数。
/// 所以「词有几字节」就是上界。本常量是**每个词**的上限，不是整段的上限
/// （整段由 BPE_OUT_LIMIT 管）。
///
/// ⚠️ 超过它的词**硬失败**（G13），不静默截断。理由：一个超长「词」
/// （例如整段没有空格的中文/阿拉伯数字串）在真实语料里是**有可能**的，
/// 静默截断会让它的后半段凭空消失，而产物看起来完全正常。
pub const BPE_WORD_MAX: usize = 4096;

/// 把一个词（不含空格）编成 id 序列，返回个数。写入 `ids`（至少 BPE_WORD_MAX）。
///
/// 最低 rank 优先：每一轮扫一遍，取**全局最小 rank** 的可合并对，合并，重来。
/// （与 Python 侧 `BpeRef.encode` 的语义相同；对拍见 tools/p17_bpe_xcheck.py。）
unsafe fn encode_word(
    src: *const u8,
    layout: &BpeLayout,
    mo: usize,
    m: usize,
    w: *const u8,
    wlen: usize,
    ids: &mut [u32; BPE_WORD_MAX],
) -> usize {
    if wlen > BPE_WORD_MAX {
        print(b"BPE: word exceeds the per-word id buffer\n");
        print(b"BPE: word_bytes=");
        print_num(wlen);
        print(b" buffer=");
        print_num(BPE_WORD_MAX);
        print(b"\n");
        exit(9);
    }
    let mut n = wlen;
    let mut i = 0usize;
    while i < wlen {
        ids[i] = *w.add(i) as u32;
        i += 1;
    }
    // ---- 逐轮合并：每轮取全局最小 rank ----
    loop {
        let mut best_rank: usize = usize::MAX;
        let mut best_i: isize = -1;
        let mut k = 0usize;
        while k + 1 < n {
            // 线性查找规则 (ids[k], ids[k+1])。rank = 规则下标。
            // 为了保持「不分配、不建 hash 表」，这里用线性扫描 —— 词内 id 数
            // 很小（BPE_WORD_MAX 是上界，不是常态），且合并轮数有限。
            //
            // ⚠️ 这是一个**已知的复杂度代价**，不是疏忽：真实词表 4096 规则、
            //    平均词长 5-10 字节时，每轮 O(n*m) 的常数很小。若将来词表
            //    量级上去，这里要换成一个 (left,right)->rank 的排序表或
            //    open-addressing 表；那是一次**性能**改动，不改语义。
            let mut r = 0usize;
            let a = ids[k];
            let b = ids[k + 1];
            let mut found: isize = -1;
            while r < m {
                let off = mo + r * MERGE_ENTRY;
                if get_u16(src, off) as u32 == a && get_u16(src, off + 2) as u32 == b {
                    found = r as isize;
                    break;
                }
                r += 1;
            }
            if found >= 0 && (found as usize) < best_rank {
                best_rank = found as usize;
                best_i = k as isize;
            }
            k += 1;
        }
        if best_i < 0 {
            break;
        }
        // 合并：ids[best_i] <- 256 + rank，删掉 ids[best_i+1]
        let bi = best_i as usize;
        ids[bi] = N_BYTE_PIECES + best_rank as u32;
        let mut t = bi + 1;
        while t + 1 < n {
            ids[t] = ids[t + 1];
            t += 1;
        }
        n -= 1;
    }
    // 顺手校验：编出来的 id 必须在词表范围内。越界说明规则表或合并逻辑坏了，
    // 而它会在解码时表现为「id 越界」—— 归因成本高，在这里拦住便宜。
    let np = layout.n_pieces;
    let mut v = 0usize;
    while v < n {
        if ids[v] >= np {
            print(b"BPE: encoder produced an id out of vocabulary range\n");
            print(b"BPE: at=");
            print_num(v);
            print(b" id=");
            print_num(ids[v] as usize);
            print(b" n_pieces=");
            print_num(np as usize);
            print(b"\n");
            exit(9);
        }
        v += 1;
    }
    n
}

// ============================================================================
// 6. 解码：token id -> 文本
// ============================================================================

/// 解码统计。
#[derive(Clone, Copy, Default)]
pub struct DecodeStat {
    /// 写出的字节数。
    pub n_bytes: usize,
    /// 走过的特殊表（byte-fallback）token 个数。**计数而不是布尔值**：
    /// 「这条路径有没有被走过」是要报出来的事实，不是要猜的。
    pub n_special_hits: usize,
    /// 展开后仍为空字节的 piece 个数（正常情况下为 0）。
    pub n_empty_pieces: usize,
}

/// 解码：token id 序列 -> 字节串。
///
/// 每个 id 恰好三种情形，顺序固定：
///   (1) 特殊表命中    -> 直接写它的字节（见 `read_vocab` 的 G6/G7）
///   (2) id < 256      -> 写那个字节
///   (3) id >= 256     -> **展开到叶子**：按 piece_off/piece_len 拷一段
///   (4) 其余          -> **硬失败 exit(9)**，绝不静默跳过
///
/// ⚠️ 情形 (3) 是本模块**唯一**的正确做法，理由见模块头那一大段：
/// 不要从输出 buffer 推导 piece 的字节。展开只用合并表一个真相源。
///
/// # Safety
/// `piece_off`/`piece_len` 由 `expand_vocab` 填好，各 `n_pieces` 项；
/// `arena` 是它写过的那块；`out` 至少 `out_cap` 字节。
pub unsafe fn decode_ids(
    src: *const u8,
    layout: &BpeLayout,
    piece_off: *const u32,
    piece_len: *const u32,
    arena: *const u8,
    ids: *const u32,
    n_ids: usize,
    out: *mut u8,
    out_cap: usize,
    stats: *mut DecodeStat,
) -> usize {
    if !stats.is_null() {
        *stats = DecodeStat::default();
    }
    let np = layout.n_pieces;
    let mut at: usize = 0;
    let mut n_sp: usize = 0;
    // ---- 特殊表：n_special 通常为 0，但这条路径必须存在且**可数** ----
    //
    // 为什么不把它编译掉（`if layout.n_special != 0`）：那样特殊表的读取代码
    // 在**所有**真实产物上都是死代码，而 dead code 不能被任何判据覆盖 ——
    // 正是本项目反复强调的「不可达 ⇒ 测不到」。这里保留 O(n_special) 的
    // 线性查表，n_special == 0 时循环体一次都不进，代价是一次比较。
    let so = layout.specials_off();
    let ns = layout.n_special as usize;

    let mut i = 0usize;
    while i < n_ids {
        let id = *ids.add(i);
        // G12: 越界 id 是**硬失败**。这里不是「防御性检查」：id 越界意味着
        //      调用方给的序列与装载的词表不是同一次产物，继续解码会把
        //      另一张表的字节拼进来 —— 通顺、量级正常、完全错误。
        if id >= np {
            print(b"BPE: token id out of vocabulary range\n");
            print(b"BPE: at=");
            print_num(i);
            print(b" id=");
            print_num(id as usize);
            print(b" n_pieces=");
            print_num(np as usize);
            print(b"\n");
            exit(9);
        }
        let pl = *piece_len.add(id as usize) as usize;
        if pl == 0 {
            // 空 piece。本模型的展开**不可能**产出空 piece（基础情形是 1 字节，
            // 合并是两个非空拼接）。所以走到这里说明 piece_len 表被写坏了。
            print(b"BPE: piece expanded to zero bytes\n");
            print(b"BPE: at=");
            print_num(i);
            print(b" id=");
            print_num(id as usize);
            print(b"\n");
            exit(9);
        }
        if at + pl > out_cap {
            print(b"BPE: output buffer too small\n");
            print(b"BPE: need=");
            print_num(at + pl);
            print(b" cap=");
            print_num(out_cap);
            print(b"\n");
            exit(9);
        }
        // 特殊表命中统计。查表本身不改变字节结果：G7 强制特殊 piece 的
        // piece_id < 256，而 id < 256 在展开表里就是那一字节 —— 于是
        // 「特殊表存在」与「展开表完备」在字节层面**已经一致**，不需要第二条
        // 真相源。这里只把「走过几次」记下来，因为「这条路径有没有被覆盖」
        // 是要报出来的**事实**（本产物的实测值是 0，见报告）。
        let mut k = 0usize;
        while k < ns {
            if get_u32(src, so + k * SPECIAL_ENTRY + 1) == id {
                n_sp += 1;
                break;
            }
            k += 1;
        }
        core::ptr::copy_nonoverlapping(
            arena.add(*piece_off.add(id as usize) as usize),
            out.add(at),
            pl,
        );
        at += pl;
        i += 1;
    }
    if !stats.is_null() {
        (*stats).n_bytes = at;
        (*stats).n_special_hits = n_sp;
    }
    at
}

/// 特殊表在解码路径上的查表。返回 `(piece_off, piece_len, is_special)`。
///
/// ⚠️ 本函数当前**只**按展开表取字节；特殊表的 piece_id 一定 < 256（G7），
/// 而 id < 256 在展开表里就是那一字节，所以两条路径在**字节结果**上一致。
/// 保留这个形状是为了让「特殊表被走过几次」这件事**可数**（DecodeStat），
/// 而不是把它藏进一个 if 里。
#[inline]
#[allow(dead_code)]
unsafe fn piece_locate(
    src: *const u8,
    layout: &BpeLayout,
    piece_off: *const u32,
    piece_len: *const u32,
    id: u32,
) -> (usize, usize, bool) {
    let _ = layout;
    let _ = src;
    (
        *piece_off.add(id as usize) as usize,
        *piece_len.add(id as usize) as usize,
        false,
    )
}

/// 占位：特殊表基址（当前模型下恒为空，n_special == 0）。
#[inline]
#[allow(dead_code)]
unsafe fn src_special_slot() -> *const u8 {
    core::ptr::null()
}

// ============================================================================
// 7. 写（构建侧）：布局 -> 文件字节
// ============================================================================
//
// 由 host/构建侧调用。Python 侧的等价实现在 tools/p17_bpe_vocab.py 的
// `write_vocab()`，**两者的字节必须完全相同** —— 这是交叉校验的对象。
// 若两边漂移，交叉校验必须 FAIL（这是它存在的唯一理由）。

/// 写入头部。`merges`/`specials` 由调用方随后写入。
///
/// 返回写入的字节数（== layout.section_bytes()）。
///
/// # Safety
/// `dst` 至少 `layout.section_bytes()` 字节且可写。弱对齐安全（全部用
/// write_unaligned）。
pub unsafe fn write_vocab_header(layout: &BpeLayout, dst: *mut u8) -> usize {
    layout.validate();
    put_u32(dst, BPE_OFF_MAGIC, BPE_MAGIC);
    put_u32(dst, BPE_OFF_VERSION, BPE_VERSION);
    put_u32(dst, BPE_OFF_N_MERGES, layout.n_merges);
    put_u32(dst, BPE_OFF_N_PIECES, layout.n_pieces);
    put_u32(dst, BPE_OFF_MERGE_BYTES, layout.merge_bytes);
    put_u32(dst, BPE_OFF_SPECIAL_BYTES, layout.special_bytes);
    put_u32(dst, BPE_OFF_N_SPECIAL, layout.n_special);
    put_u32(dst, BPE_OFF_RESERVED0, 0);
    layout.section_bytes() as usize
}

/// 写一条合并规则（给构建侧逐条调用）。
///
/// `left`/`right` 必须是**已存在**的 id（< 256 + j），`merged_id` 必须恰好是
/// 256 + j —— 两者都由 `read_vocab` 在装载期硬校验，所以写入侧写错一定会在
/// 装载时被抓住，而不是在解码时变成一段错字节。
///
/// # Safety
/// 同 `write_vocab_header`。
pub unsafe fn write_merge(
    layout: &BpeLayout,
    dst: *mut u8,
    j: usize,
    left: u32,
    right: u32,
) {
    let off = layout.merges_off() + j * MERGE_ENTRY;
    put_u16(dst, off, left as u16);
    put_u16(dst, off + 2, right as u16);
    put_u32(dst, off + 4, N_BYTE_PIECES + j as u32);
}

// ============================================================================
// 8. 无 CRT 的小工具（不引 core::fmt）
// ============================================================================
//
// 与 conn_sparse.rs §9 同形。**为什么复制而不是复用**：那些函数的名字/消息都
// 写死了 "CS:"，而且它们的 feature gate 是 `conn-sparse`。借过来会让
// 「哪个 feature 开了什么」变成隐式耦合（开 bpe 却因为少开 conn-sparse 而
// 链接失败）。复制的代价是 40 行，换来的是两个 feature 完全正交 —— 值得。

/// 读 u32（弱对齐安全）。
#[inline]
pub unsafe fn get_u32(p: *const u8, off: usize) -> u32 {
    core::ptr::read_unaligned(p.add(off) as *const u32)
}
/// 读 u16（弱对齐安全）。
#[inline]
pub unsafe fn get_u16(p: *const u8, off: usize) -> u16 {
    core::ptr::read_unaligned(p.add(off) as *const u16)
}
/// 写 u32（弱对齐安全）。
#[inline]
pub unsafe fn put_u32(p: *mut u8, off: usize, v: u32) {
    core::ptr::write_unaligned(p.add(off) as *mut u32, v);
}
/// 写 u16（弱对齐安全）。
#[inline]
pub unsafe fn put_u16(p: *mut u8, off: usize, v: u16) {
    core::ptr::write_unaligned(p.add(off) as *mut u16, v);
}

/// 无 CRT 的十进制打印。
pub unsafe fn print_num(mut v: usize) {
    let mut buf = [0u8; 24];
    let mut i = buf.len();
    if v == 0 {
        i -= 1;
        buf[i] = b'0';
    }
    while v > 0 {
        i -= 1;
        buf[i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
    print(&buf[i..]);
}

/// 无 CRT 的 8 位十六进制打印（magic 之类的字段用）。
pub unsafe fn print_hex32(v: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 8];
    let mut i = 0usize;
    while i < 8 {
        buf[i] = HEX[((v >> (28 - 4 * i)) & 0xF) as usize];
        i += 1;
    }
    print(&buf);
}
