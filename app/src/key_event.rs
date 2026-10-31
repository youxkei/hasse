/// A push-to-talk key transition, shared by both platform hotkeys.
#[derive(Debug, Clone, Copy)]
pub enum KeyEvent {
    Pressed,
    Released,
}
