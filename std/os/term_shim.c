// std/os/term_shim.c — Zero-Rust POSIX Terminal Raw Mode, Signal-Safe Restoration & Non-Blocking Key Input
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <termios.h>
#include <sys/select.h>

static struct termios g_orig_termios;
static volatile sig_atomic_t g_raw_active = 0;

static void write_all_raw(const char* s, size_t len) {
    while (len > 0) {
        ssize_t w = write(STDOUT_FILENO, s, len);
        if (w <= 0) break;
        s += (size_t)w;
        len -= (size_t)w;
    }
}

static void restore_term_signal_safe(void) {
    if (!g_raw_active) return;
    g_raw_active = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
    // Exit alternate screen FIRST (\x1b[?1049l), THEN explicitly show cursor (\x1b[?25h) & reset SGR (\x1b[0m)
    // on the main screen buffer (also reset alt screen cursor beforehand for terminals that save DECTCEM per buffer).
    static const char kRestoreSeq[] = "\x1b[0m\x1b[?25h\x1b[?1049l\x1b[?25h\x1b[0m";
    write_all_raw(kRestoreSeq, sizeof(kRestoreSeq) - 1);
}

static void restore_term_atexit(void) {
    restore_term_signal_safe();
}

static void handle_term_signal(int sig) {
    restore_term_signal_safe();
    signal(sig, SIG_DFL);
    raise(sig);
}

int64_t tn_term_enter_raw(void) {
    if (!isatty(STDIN_FILENO)) return 0;
    if (tcgetattr(STDIN_FILENO, &g_orig_termios) != 0) return 0;
    struct termios raw = g_orig_termios;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    g_raw_active = 1;

    static int handlers_installed = 0;
    if (!handlers_installed) {
        handlers_installed = 1;
        atexit(restore_term_atexit);
        struct sigaction sa;
        sa.sa_handler = handle_term_signal;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGTERM, &sa, NULL);
        sigaction(SIGQUIT, &sa, NULL);
        sigaction(SIGHUP, &sa, NULL);
    }

    // Switch to alternate screen buffer, hide cursor, clear & home
    static const char kEnterSeq[] = "\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H";
    write_all_raw(kEnterSeq, sizeof(kEnterSeq) - 1);
    return 1;
}

int64_t tn_term_leave_raw(void) {
    fflush(stdout);
    restore_term_signal_safe();
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
