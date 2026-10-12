# PATH GUARD: 严格隔离路径 A（任务书第七部分）
#
# ASCII-ONLY BY DESIGN: PowerShell 5.1 reads .ps1 as GBK/ANSI unless the file
# has a UTF-8 BOM. Chinese text without a BOM gets mis-decoded and can produce
# stray quote/brace tokens -> parse errors pointing at innocent lines.
# So this file's CODE is pure ASCII; all explanation is in English.
#
# WHY THIS EXISTS
#   The task brief names "path A" (old sparse = dense matrix + runtime skip) and
#   forbids it on any production path. A rule that is only written in a document
#   is not a rule -- it decays. This makes it executable.
#
# WHAT IT CHECKS
#   G1  production paths must not reference `ref_sparse_a`
#   G2  production paths must not import the path-A storage module
#       (vmlm_core::sparse / crate::sparse) -- that module implements
#       gather_rows / mask_from_thresh / gemm_nt_cols_selected, i.e. the
#       dense+skip formulation.
#   G3  production paths must not contain path-A VOCABULARY for pools
#       (the "shared pool" idea was measured and RETRACTED: docs/P18-B).
#   G4  every sparse_b / trans source file carries a `// PATH:` header line.
#
# DELIBERATELY NOT BANNED (and why -- this is the honest part)
#   `compute_act_mask` is NOT a path-A word. Measured: it selects top-k over
#   k0=20 connections per neuron (core/src/conn_train.rs:479 `for j in 0..k0`),
#   not over 1536. That is the B2 formulation, and B2 needs a selection step by
#   definition. Banning it would ban B2. See ref_sparse_a/AUDIT_PATH_ISOLATION.md
#   section 3.
#
# USAGE: powershell -NoProfile -ExecutionPolicy Bypass -File tools/ci_path_guard.ps1
# EXIT : 0 = clean, 1 = violation(s) found

$ErrorActionPreference = "Continue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$script:violations = 0
$script:checks = 0

function Fail($rule, $where, $detail) {
    $script:violations++
    Write-Host ("  [FAIL] " + $rule + "  @ " + $where) -ForegroundColor Red
    Write-Host ("         " + $detail)
}

function Pass($rule, $detail) {
    $script:checks++
    Write-Host ("  [ok]   " + $rule + "  " + $detail) -ForegroundColor Green
}

# Production paths.
#
# CORRECTED LIST (finding from b1-builder, verified by the Lead):
#   The first version listed "engine" and "trainer". NEITHER EXISTS in this
#   workspace: `engine/` is an empty directory and `trainer/` was never created.
#   So G1 was silently scanning two empty trees and reporting a confident
#   "0 hits" -- the exact "check that cannot fail" class this project keeps
#   running into. The real workspace members are declared in the root
#   Cargo.toml: core, asm, pt2vmb, runtime, tests, train, server.
#
#   The trees below are those members, plus the two task-brief directory names
#   (engine/ trainer/) retained so that if they are ever populated they are
#   covered too.
$prodDirs = @("core", "runtime", "server", "asm", "pt2vmb", "train", "tests", "engine", "trainer")

# Emptiness check: a production tree that does not exist or holds no source
# files cannot be "clean" -- it is UNSCANNED, and reporting "ok" would be a lie.
$scanned = @()
$unscanned = @()
foreach ($d in $prodDirs) {
    if (-not (Test-Path $d)) { $unscanned += $d; continue }
    $any = Get-ChildItem -Path $d -Recurse -File -Include *.rs -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($null -eq $any) { $unscanned += $d } else { $scanned += $d }
}

Write-Host "===== PATH GUARD: path-A isolation =====" -ForegroundColor Yellow
Write-Host ("scanned production trees : " + ($scanned -join ", "))
Write-Host ("not present / no .rs     : " + ($unscanned -join ", "))

# ---------------------------------------------------------------------------
# Comment stripping.
#
# WHY THIS IS NECESSARY (finding from b1-builder, reproduced by the Lead):
#   G1 matched the literal text "ref_sparse_a" with -SimpleMatch, which does not
#   skip comments. A file whose HEADER says "PATH: DENSE -- does not reference
#   ref_sparse_a" therefore tripped its own guard. Two real files failed this
#   way while containing zero actual path-A imports.
#
# The fix strips `//` line comments and `#` (PowerShell/Python) comments before
# matching, so the guard tests CODE, not prose. This is deliberately done here
# rather than by rewording the comments: a guard that forbids naming the thing
# it forbids makes honest documentation impossible, and that documentation is
# what lets a reader verify the isolation.
# ---------------------------------------------------------------------------
function Get-CodeLines($path, $ext) {
    $out = @()
    $n = 0
    foreach ($ln in (Get-Content $path -ErrorAction SilentlyContinue)) {
        $n++
        $code = $ln
        if ($ext -eq ".ps1" -or $ext -eq ".py") {
            # strip full-line # comments only; a trailing # is left alone
            # because it also appears inside strings (colour codes, etc.)
            $t = $ln.TrimStart()
            if ($t.StartsWith("#")) { $code = "" }
        } else {
            # Rust / TOML: strip from the first // to end of line. Block
            # comments are not used for these headers anywhere in this repo.
            $i = $ln.IndexOf("//")
            if ($i -ge 0) { $code = $ln.Substring(0, $i) }
        }
        if ($code.Trim() -ne "") { $out += [pscustomobject]@{ Line = $code; Num = $n } }
    }
    return $out
}

function Scan-Code($dir, $patterns, $simple) {
    $hits = @()
    $files = Get-ChildItem -Path $dir -Recurse -File -Include *.rs, *.ps1, *.py, *.toml -ErrorAction SilentlyContinue
    foreach ($f in $files) {
        $ext = $f.Extension.ToLower()
        foreach ($cl in (Get-CodeLines $f.FullName $ext)) {
            foreach ($p in $patterns) {
                $m = $false
                if ($simple) { $m = $cl.Line.Contains($p) }
                else { $m = ($cl.Line -match $p) }
                if ($m) {
                    $hits += [pscustomobject]@{ Path = $f.FullName; LineNumber = $cl.Num; Line = $cl.Line.Trim() }
                    break
                }
            }
        }
    }
    return $hits
}

# ---------------------------------------------------------------------------
# G1: no production reference to ref_sparse_a
# ---------------------------------------------------------------------------
$g1 = 0
foreach ($d in $scanned) {
    $hits = Scan-Code $d @("ref_sparse_a") $true
    foreach ($h in $hits) {
        Fail "G1" ($h.Path + ":" + $h.LineNumber) $h.Line
        $g1++
    }
}
if ($g1 -eq 0) { Pass "G1 no ref_sparse_a reference on production paths" (($scanned.Count).ToString() + " non-empty trees scanned") }

# ---------------------------------------------------------------------------
# G2: no ENGINE import of the path-A sparse storage module.
#     `core/src/sparse.rs` IS path A (gather_rows, mask_from_thresh,
#     gemm_nt_cols_selected). Importing it from runtime/ means the engine would
#     link the dense+skip formulation.
#
#     NOTE the deliberate asymmetry: core and train are in $prodDirs for G1 but
#     NOT here, because core DEFINES the module and train may reference it as
#     historical/comparison scaffolding. G2 is specifically about the ENGINE
#     (runtime/server) not linking dense+skip code.
# ---------------------------------------------------------------------------
$g2 = 0
$g2targets = @("runtime", "server", "engine", "trainer")
foreach ($d in $g2targets) {
    if (-not (Test-Path $d)) { continue }
    $hits = Scan-Code $d @("vmlm_core::sparse", "crate::sparse") $true
    foreach ($h in $hits) {
        Fail "G2" ($h.Path + ":" + $h.LineNumber) $h.Line
        $g2++
    }
}
if ($g2 -eq 0) { Pass "G2 engine/server link no path-A sparse module" "0 hits" }

# ---------------------------------------------------------------------------
# G3: path-A POOL vocabulary. The "shared source pool" 8.62x claim was measured
#     and retracted (docs/P18-B section 1). The word must not reappear on a
#     production path, because it is the exact concept that was withdrawn.
# ---------------------------------------------------------------------------
$g3words = @(
    [char]0x5171 + [char]0x4EAB + [char]0x6C60,   # shared pool (CJK)
    "candidate_pool",
    "shared_pool"
)
$g3 = 0
foreach ($d in $scanned) {
    $hits = Scan-Code $d $g3words $true
    foreach ($h in $hits) {
        Fail "G3" ($h.Path + ":" + $h.LineNumber) $h.Line
        $g3++
    }
}
if ($g3 -eq 0) { Pass "G3 no path-A pool vocabulary on production paths" (($scanned.Count).ToString() + " non-empty trees scanned") }

# ---------------------------------------------------------------------------
# G4: `// PATH:` header on every sparse_b / trans source file.
#     The brief requires it (hard constraint 14); an unlabelled file is how the
#     two architectures get confused in the first place.
# ---------------------------------------------------------------------------
$g4 = 0
$labelled = 0
$scan = @()
foreach ($d in @("sparse_b", "trans")) {
    if (Test-Path $d) {
        $scan += Get-ChildItem -Path $d -Recurse -File -Include *.rs -ErrorAction SilentlyContinue
    }
}
foreach ($f in $scan) {
    if ($f.Length -eq 0) { continue }
    # Read first 40 lines; the header may sit after a license/blank block.
    $head = Get-Content $f.FullName -TotalCount 40 -ErrorAction SilentlyContinue
    $hasTag = $false
    foreach ($ln in $head) {
        if ($ln -match "^//\s*PATH:\s*(B1|B2|A|DENSE)") { $hasTag = $true; break }
    }
    if (-not $hasTag) {
        Fail "G4" $f.FullName "missing `// PATH: B1|B2|A|DENSE` header in first 40 lines"
        $g4++
    } else {
        $labelled++
    }
}
if ($g4 -eq 0) { Pass "G4 every sparse_b/trans source carries a PATH header" ($labelled.ToString() + " files") }

# ---------------------------------------------------------------------------
# G5: the ACTIVE path tag must not be "A" anywhere under sparse_b/ or trans/.
#     A file whose header says `// PATH: A` inside the production tree is a
#     contradiction: path A is not a production path.
# ---------------------------------------------------------------------------
$g5 = 0
foreach ($f in $scan) {
    if ($f.Length -eq 0) { continue }
    $head = Get-Content $f.FullName -TotalCount 40 -ErrorAction SilentlyContinue
    foreach ($ln in $head) {
        if ($ln -match "^//\s*PATH:\s*A\s*$") {
            Fail "G5" $f.FullName "path-A tagged file inside sparse_b/ or trans/"
            $g5++
            break
        }
    }
}
if ($g5 -eq 0) { Pass "G5 no path-A tagged file under sparse_b/trans" "0 hits" }

# ---------------------------------------------------------------------------
# G6: SELF-TEST -- the guard must be able to FAIL.
#
# WHY: after the fix above, G1/G2/G3 all reported "clean". A guard that reports
# clean and a guard that is broken look identical from the outside. This project
# has a standing rule that every criterion ships with a can-fail control
# (tools/p16_sparse_ops.py section D, the P18 kernel checks, the gradient check
# under a frozen mask -- all of them do this).
#
# The test plants a REAL path-A import in a THROWAWAY file inside a scanned
# tree, confirms the scanner reports it, then deletes the file. It uses a
# subdirectory that is removed afterwards, so the workspace is left untouched.
# ---------------------------------------------------------------------------
$g6 = 0
$probeDir = Join-Path $root "core/src/_guard_selftest"
try {
    New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
    $probeFile = Join-Path $probeDir "canfail.rs"
    # A REAL code line (not a comment) that G1 and G2 must both catch.
    Set-Content -Path $probeFile -Value "use crate::sparse::gather_rows;" -Encoding ASCII
    Add-Content -Path $probeFile -Value "// ref_sparse_a is named only in this comment"
    $codeHits = Scan-Code "core" @("crate::sparse") $true
    $commentHits = Scan-Code "core" @("ref_sparse_a") $true
    $codeHitInProbe = @($codeHits | Where-Object { $_.Path -like "*_guard_selftest*" }).Count
    $commentHitInProbe = @($commentHits | Where-Object { $_.Path -like "*_guard_selftest*" }).Count
    if ($codeHitInProbe -gt 0) {
        Pass "G6a can-fail: a real code import IS detected" ("hits=" + $codeHitInProbe)
    } else {
        Fail "G6a can-fail" "probe file" "a real code import was NOT detected -- guard is broken"
        $g6++
    }
    if ($commentHitInProbe -eq 0) {
        Pass "G6b comment stripping: a comment naming the token is NOT a violation" "0 hits (by design)"
    } else {
        Fail "G6b comment-strip" "probe file" "comment text was still counted as a violation"
        $g6++
    }
} finally {
    if (Test-Path $probeDir) { Remove-Item -Recurse -Force $probeDir -ErrorAction SilentlyContinue }
}
if (Test-Path $probeDir) {
    Fail "G6c cleanup" $probeDir "self-test directory could not be removed"
    $g6++
} else {
    Pass "G6c cleanup: self-test directory removed" "workspace untouched"
}

Write-Host ""
Write-Host ("checks passed: " + $script:checks + "   violations: " + $script:violations)
if ($script:violations -gt 0) {
    Write-Host "PATH GUARD: FAIL" -ForegroundColor Red
    exit 1
}
Write-Host "PATH GUARD: CLEAN" -ForegroundColor Green
exit 0
