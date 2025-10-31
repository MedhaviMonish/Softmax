#pragma once
#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <sstream>

enum LogLevel { INFO, WARNING, ERROR };

class Logger {
public:
  static void log(LogLevel level, const std::string &msg, const char *file,
                  int line) {
    std::string levelStr;
    switch (level) {
    case INFO:
      levelStr = "INFO";
      break;
    case WARNING:
      levelStr = "WARN";
      break;
    case ERROR:
      levelStr = "ERROR";
      break;
    }

    std::cerr << "[" << timestamp() << "] " << levelStr << " in " << file << ":"
              << line << " -> " << msg << std::endl;

    if (level == ERROR) {
      std::abort();
    }
  }

private:
  static std::string timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
    return std::string(buf);
  }
};

#define LOG_INFO(msg) Logger::log(INFO, msg, __FILE__, __LINE__)
#define LOG_WARN(msg) Logger::log(WARNING, msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) Logger::log(ERROR, msg, __FILE__, __LINE__)
