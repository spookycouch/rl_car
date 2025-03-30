#ifndef MOTOR_H
#define MOTOR_H

#include <PID_v1.h>

#define PID_MIN_OUTPUT -127.0
#define PID_MAX_OUTPUT 127.0

class Motor {
  private:
    PID* pid_controller;
    // PID gains
    double pid_k_proportional;
    double pid_k_integral;
    double pid_k_derivative;

    // motor output pins
    int gpio_pwm;
    int gpio_dir;
    int gpio_slp;

    double pid_output = 0;

    // velocity measurements
    double velocity_rad = 0;
    double target_velocity_rad = 0;
    double max_velocity_rad;


    Encoder* encoder;
    // number of encoder counts for one revolution of the wheel
    unsigned long encoder_cpr;
    // encoder measurements
    long prev_encoder_position = 0;
    unsigned long prev_time_ms = millis();
  
  public:
    Motor(
      double pid_k_proportional,
      double pid_k_integral,
      double pid_k_derivative,
      int gpio_pwm,
      int gpio_dir,
      int gpio_slp,
      Encoder* encoder,
      unsigned long encoder_cpr,
      double max_velocity_rad
    ) {
      this->pid_controller = new PID(
        &velocity_rad,
        &pid_output,
        &target_velocity_rad,
        pid_k_proportional,
        pid_k_integral,
        pid_k_derivative,
        DIRECT
      );
      pid_controller->SetMode(AUTOMATIC);
      pid_controller->SetOutputLimits(PID_MIN_OUTPUT, PID_MAX_OUTPUT);

      this->gpio_pwm = gpio_pwm;
      this->gpio_dir = gpio_dir;
      this->gpio_slp = gpio_slp;
      pinMode(gpio_pwm, OUTPUT);
      pinMode(gpio_dir, OUTPUT);
      pinMode(gpio_slp, OUTPUT);

      this->pid_output = 0;

      this->encoder = encoder;
      this->encoder_cpr = encoder_cpr;
      this->max_velocity_rad = max_velocity_rad;

      prev_encoder_position = encoder->read();
      this->prev_time_ms = millis();
    }

    double get_target_velocity() {
      return this->target_velocity_rad;
    } 

    void set_target_velocity(double velocity_rad) {
      this->target_velocity_rad = std::clamp(velocity_rad, -max_velocity_rad, max_velocity_rad);
    }

    double get_velocity() {
      return this->velocity_rad;
    }

    void update() {
      if (this->target_velocity_rad == 0) {
        pid_controller->SetMode(0);
        pid_output = 0;
      } else if (pid_controller->GetMode() == 0) {
        pid_controller->SetMode(1);
      }

      long delta_position = encoder->read() - prev_encoder_position;
      float delta_position_rad = ((float)delta_position/encoder_cpr) * 2 * M_PI;
      float delta_time_s = (millis() - prev_time_ms) / 1000.0;
      if (delta_time_s > 0) {
        velocity_rad = delta_position_rad/delta_time_s;
      } else {
        velocity_rad = 0;
      }

      pid_controller->Compute();

      // set direction
      if (pid_output < 0) {
        digitalWrite(gpio_dir, HIGH);
      }
      else {
        digitalWrite(gpio_dir, LOW);
      }

      // set sleep pin
      digitalWrite(gpio_slp, HIGH);
      
      // set pwm output
      analogWrite(gpio_pwm, abs(pid_output));

      encoder->write(0);
      prev_encoder_position = encoder->read();
      prev_time_ms = millis();
    }
};

#endif
