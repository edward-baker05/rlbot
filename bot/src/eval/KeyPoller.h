#pragma once

#include <termios.h>

namespace Dash {

// Single-keypress terminal input, for spectate's --rewards mode.
//
// Puts stdin into non-canonical no-echo mode so a key is readable the instant
// it is pressed, without waiting for Enter. ISIG is left on, so Ctrl-C keeps
// working normally.
//
// The terminal is a process-wide resource that outlives us if we mishandle it:
// a spectator killed in raw mode leaves the user's shell with no echo. So the
// mode is restored from the destructor AND from SIGINT/SIGTERM handlers, since
// scripts/spectate.sh terminates the bot with SIGTERM when the visualizer
// window closes. Only one instance may exist at a time.
//
// When stdin is not a terminal (piped, or the spectator was launched without
// scripts/spectate.sh wiring up /dev/tty) this degrades to an inert object:
// Active() is false and Poll() always returns 0. Callers should treat that as
// "no key was pressed" and keep running, not as an error.
class KeyPoller {
  public:
	KeyPoller();
	~KeyPoller();

	KeyPoller(const KeyPoller &) = delete;
	KeyPoller &operator=(const KeyPoller &) = delete;

	// False when stdin is not a terminal; Poll() then never returns a key.
	bool Active() const { return active; }

	// The next buffered keypress, or 0 when none is waiting. Never blocks.
	char Poll();

  private:
	bool active = false;
	termios saved = {};
};

} // namespace Dash
