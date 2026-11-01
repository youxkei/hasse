//! Remembering which application had the keyboard, so it can get it back after
//! the overlay borrowed the focus for editing.
//!
//! The review overlay is the only time hasse takes focus. Without this the
//! inserted text would land in the overlay instead of the app the user was
//! typing in.

#[cfg(windows)]
mod imp {
    use windows::Win32::Foundation::HWND;
    use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, SetForegroundWindow};

    pub struct Focus(isize);

    pub fn capture() -> Option<Focus> {
        let window = unsafe { GetForegroundWindow() };
        if window.is_invalid() {
            None
        } else {
            Some(Focus(window.0 as isize))
        }
    }

    pub fn restore(focus: &Focus) {
        unsafe {
            let _ = SetForegroundWindow(HWND(focus.0 as *mut core::ffi::c_void));
        }
    }

    /// The work area of the monitor that owns `focus`. The rectangle is in the
    /// same virtualized pixels `SetWindowPos` takes for this process.
    pub fn monitor_rect(focus: &Focus) -> Option<[f32; 4]> {
        use windows::Win32::Graphics::Gdi::{
            GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONEAREST,
        };

        unsafe {
            let monitor = MonitorFromWindow(
                HWND(focus.0 as *mut core::ffi::c_void),
                MONITOR_DEFAULTTONEAREST,
            );
            if monitor.is_invalid() {
                return None;
            }
            let mut info = MONITORINFO {
                cbSize: std::mem::size_of::<MONITORINFO>() as u32,
                ..Default::default()
            };
            if !GetMonitorInfoW(monitor, &mut info).as_bool() {
                return None;
            }
            let rect = info.rcWork;
            Some([
                rect.left as f32,
                rect.top as f32,
                (rect.right - rect.left) as f32,
                (rect.bottom - rect.top) as f32,
            ])
        }
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2_app_kit::{NSApplicationActivationOptions, NSRunningApplication, NSWorkspace};

    pub struct Focus(i32);

    pub fn capture() -> Option<Focus> {
        let workspace = NSWorkspace::sharedWorkspace();
        let app = workspace.frontmostApplication()?;
        Some(Focus(app.processIdentifier()))
    }

    /// `NSRunningApplication` has no replacement for this call; the option is
    /// deprecated on macOS 14 but still activates the app.
    #[allow(deprecated)]
    pub fn restore(focus: &Focus) {
        if let Some(app) = NSRunningApplication::runningApplicationWithProcessIdentifier(focus.0) {
            let _ = app.activateWithOptions(NSApplicationActivationOptions::ActivateIgnoringOtherApps);
        }
    }

    /// The overlay stays on the screen it is already on.
    pub fn monitor_rect(_focus: &Focus) -> Option<[f32; 4]> {
        None
    }
}

pub use imp::{Focus, capture, monitor_rect, restore};
