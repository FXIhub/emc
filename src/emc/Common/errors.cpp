/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
*   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
*/

#include <stdarg.h>
#include <errors.h>

void error_exit_with_message(const char *message, ...) {
  va_list ap;
  va_start(ap, message);
  fprintf(stderr, "Error: ");
  vfprintf(stderr, message, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

void error_warning(const char *message, ...) {
  va_list ap;
  va_start(ap, message);
  fprintf(stderr, "Warning: ");
  vfprintf(stderr, message, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

/* Capture a crtl-c event to make a final iteration be
   run with the individual masked used in the compression.
   This is consistent with the final iteration when not
   interupted. Ctrl-c again will exit immediatley. */
void nice_exit(int quit_requested) {
  if (quit_requested == 0) {
    quit_requested = 1;
  } else {
    exit(1);
  }
}
