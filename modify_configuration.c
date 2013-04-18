#include <libconfig.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  char *input_file;
  char *output_file;
  char help_text[] = 
    "Usage modify_configuration INPUT.conf OUTPUT.conf [OPTION1 VALUE1 OPTION2 VALUE2 ...]\n";
  int c;
  int opterr = 0;

  while((c = getopt(argc,argv,"h")) != -1) {
    if (c == -1) {
      break;
    }
    switch(c) {
    case('h'):
      printf("%s\n",help_text);
      exit(0);
      break;
    }
  }

  if (argc-optind-1 < 2) {
    printf("Too few arguments, must give at least input and output file\n");
    exit(1);
  }

  input_file = argv[optind];
  output_file = argv[optind+1];

  /* read config file */
  config_t config;
  config_init(&config);
  if (!config_read_file(&config, input_file)) {
    printf("Error reading %s\n", input_file);
    exit(1);
  }

  char *key, *value;
  config_setting_t *config_setting;
  for (int i = optind+2; i < argc; i+=2) {
    //sprintf(argv[i], "%s=%s", key, value);
    key = argv[i];
    value = argv[i+1];
    config_setting = config_lookup(&config, key);
    //printf("Type = %d\n", config_setting_get_format(config_setting));
    //printf("Type = %d\n", config_setting->format);
    //exit(0);
    char *endptr;
    long v_int = strtol(value, &endptr, 10);
    if (endptr[0] == '\0') {
      config_setting_set_int(config_setting, v_int);
      continue;
    }
    float v_float = strtof(value, &endptr);
    if (endptr[0] == '\0') {
      config_setting_set_float(config_setting, v_float);
      continue;
    }
    if (strcmp(value, "true")) {
      config_setting_set_bool(config_setting, 1);
      continue;
    }
    if (strcmp(value, "false")) {
      config_setting_set_bool(config_setting, 0);
      continue;
    }
    config_setting_set_string(config_setting, value);
  }
  
  if (!config_write_file(&config, output_file)) {
    printf("Error writing %s\n", output_file);
    exit(1);
  }
  config_destroy(&config);
}
