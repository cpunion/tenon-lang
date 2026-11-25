// std/os/term_shim.c — Tiny Zero-Rust POSIX Terminal Raw Mode & Non-Blocking Key Input Helper
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/select.h>

static struct termios g_orig_termios;
static int g_raw_active = 0;

int64_t tn_term_enter_raw(void) {
    if (!isatty(STDIN_FILENO)) return 0;
    if (tcgetattr(STDIN_FILENO, &g_orig_termios) != 0) return 0;
    struct termios raw = g_orig_termios;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    g_raw_active = 1;
    // Switch to alternate screen buffer & hide cursor
    fputs("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H", stdout);
    fflush(stdout);
    return 1;
}

int64_t tn_term_leave_raw(void) {
    if (g_raw_active) {
        tcsetattr(STDIN_FILENO, TCSANOW, &g_orig_termios);
        g_raw_active = 0;
        // Restore cursor & main screen buffer
        fputs("\x1b[0m\x1b[?25h\x1b[?1049l", stdout);
        fflush(stdout);
    }
    return 1;
}

int64_t tn_term_poll_key(int64_t timeout_ms) {
    if (!isatty(STDIN_FILENO)) {
        if (timeout_ms > 0) usleep((useconds_t)(timeout_ms * 1000));
        return -1;
    }
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (suseconds_t)((timeout_ms % 1000) * 1000);
    int r = select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv);
    if (r > 0) {
        unsigned char ch = 0;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            return (int64_t)ch;
        }
    }
    return -1;
}

int64_t tn_term_flush(void) {
    fflush(stdout);
    return 1;
}
