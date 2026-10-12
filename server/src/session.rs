//! Conversation state shared by the CLI (and available to the server).

use crate::chat::Message;

#[derive(Debug, Clone)]
pub struct Session {
    pub model: String,
    pub system: Option<String>,
    pub messages: Vec<Message>,
}

impl Session {
    pub fn new(model: impl Into<String>) -> Session {
        Session {
            model: model.into(),
            system: None,
            messages: Vec::new(),
        }
    }

    pub fn set_system(&mut self, s: impl Into<String>) {
        self.system = Some(s.into());
    }

    pub fn push_user(&mut self, content: impl Into<String>) {
        self.messages.push(Message {
            role: "user".into(),
            content: content.into(),
        });
    }

    pub fn push_assistant(&mut self, content: impl Into<String>) {
        self.messages.push(Message {
            role: "assistant".into(),
            content: content.into(),
        });
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Messages as they would be sent on the wire, with the system prompt first.
    pub fn wire_messages(&self) -> Vec<Message> {
        let mut v = Vec::with_capacity(self.messages.len() + 1);
        if let Some(s) = &self.system {
            v.push(Message {
                role: "system".into(),
                content: s.clone(),
            });
        }
        v.extend(self.messages.iter().cloned());
        v
    }

    /// Number of characters currently held (a proxy for context size; the real
    /// limit lives in whatever model is loaded).
    pub fn context_chars(&self) -> usize {
        self.wire_messages().iter().map(|m| m.content.chars().count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_message_comes_first() {
        let mut s = Session::new("m");
        s.push_user("u1");
        s.set_system("sys");
        s.push_assistant("a1");
        let w = s.wire_messages();
        assert_eq!(w.len(), 3);
        assert_eq!(w[0].role, "system");
        assert_eq!(w[1].role, "user");
        assert_eq!(w[2].role, "assistant");
        assert_eq!(s.context_chars(), 3 + 2 + 2);
        s.clear();
        assert_eq!(s.wire_messages().len(), 1); // system only
    }
}
