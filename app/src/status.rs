/// What the overlay shows. Shared by both platform overlays.
#[derive(Clone, Debug)]
pub enum Status {
    Starting,
    Recording,
    Transcribing,
    Error(String),
}
