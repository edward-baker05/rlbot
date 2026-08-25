#include "KeyPoller.h"

#include <csignal>
#include <unistd.h>

namespace Dash {

namespace {

// Signal handlers cannot take arguments, so the state they need to undo lives
// here. Written only by the KeyPoller constructor/destructor, which run before
// any handler is installed and after it is removed.
volatile sig_atomic_t g_rawModeActive = 0;
termios g_savedTermios = {};

void (*g_prevSigint)(int) = nullptr;
void (*g_prevSigterm)(int) = nullptr;

// tcsetattr is async-signal-safe, so this is legal from a handler.
void RestoreTermios() {
	if (g_rawModeActive) {
		g_rawModeActive = 0;
		tcsetattr(STDIN_FILENO, TCSANOW, &g_savedTermios);
	}
}

void HandleFatalSignal(int sig) {
	RestoreTermios();

	// Re-raise with the default disposition so the exit status still reports
	// the signal, and so the parent script's own trap behaves as it expects.
	std::signal(sig, SIG_DFL);
	std::raise(sig);
}

} // namespace

KeyPoller::KeyPoller() {
	if (!isatty(STDIN_FILENO))
		return;

	if (tcgetattr(STDIN_FILENO, &saved) != 0)
		return;

	termios raw = saved;
	// ~ICANON delivers keys without waiting for Enter; ~ECHO stops them being
	// painted over the reward tables. ISIG is deliberately left set.
	raw.c_lflag &= ~(ICANON | ECHO);
	// VMIN 0 / VTIME 0 makes read() return immediately with whatever is
	// buffered, which is what makes Poll() non-blocking.
	raw.c_cc[VMIN] = 0;
	raw.c_cc[VTIME] = 0;

	if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0)
		return;

	active = true;
	g_savedTermios = saved;
	g_rawModeActive = 1;
	g_prevSigint = std::signal(SIGINT, HandleFatalSignal);
	g_prevSigterm = std::signal(SIGTERM, HandleFatalSignal);
}

KeyPoller::~KeyPoller() {
	if (!active)
		return;

	std::signal(SIGINT, g_prevSigint ? g_prevSigint : SIG_DFL);
	std::signal(SIGTERM, g_prevSigterm ? g_prevSigterm : SIG_DFL);
	RestoreTermios();
	active = false;
}

char KeyPoller::Poll() {
	if (!active)
		return 0;

	char c = 0;
	if (read(STDIN_FILENO, &c, 1) != 1)
		return 0;

	return c;
}

} // namespace Dash
