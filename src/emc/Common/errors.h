/*  Copyright 2014-now The EMC Project Authors. All Rights Reserved.
 *   First commit by Jing Liu (jing.liu@it.uu.se /jing.liu@icm.uu.se).
 */
#ifndef ERRORS_H
#define ERRORS_H

#include <stdio.h>
#include <emc_common.h>

    void nice_exit(int sig);
    void error_warning(const char *message, ...);
    void error_exit_with_message(const char *message, ...);
#endif
