#include <Encoder.h>
#include "motor.h"
 
#define LEFT_ENC_A 5
#define LEFT_ENC_B 6
#define LEFT_PWM 11
#define LEFT_DIR 10
#define LEFT_SLP 8

#define RIGHT_ENC_A 21
#define RIGHT_ENC_B 22
#define RIGHT_PWM 20
#define RIGHT_DIR 19
#define RIGHT_SLP 18

double pid_k_proportional = 1.0;
double pid_k_integral = 35;
double pid_k_derivative = 0.1;
long encoder_cpr = 1440;
double max_velocity_rad = 2 * M_PI;

Motor* left_motor;
Motor* right_motor;
Encoder* left_encoder;
Encoder* right_encoder;


void setup () {
  left_encoder = new Encoder(LEFT_ENC_A, LEFT_ENC_B);
  right_encoder = new Encoder(RIGHT_ENC_A, RIGHT_ENC_B);
  left_encoder->write(0);
  right_encoder->write(0);
  
  left_motor = new Motor(
    pid_k_proportional,
    pid_k_integral,
    pid_k_derivative,
    LEFT_PWM,
    LEFT_DIR,
    LEFT_SLP,
    left_encoder,
    encoder_cpr,
    max_velocity_rad
  );

  right_motor = new Motor(
    pid_k_proportional,
    pid_k_integral,
    pid_k_derivative,
    RIGHT_PWM,
    RIGHT_DIR,
    RIGHT_SLP,
    right_encoder,
    encoder_cpr,
    max_velocity_rad
  );

  Serial.begin(115200);
}

void loop () {
  double target_velocity_rad = sin(millis()/1000.0) * M_PI;
  left_motor->set_target_velocity(target_velocity_rad);
  left_motor->update();
  right_motor->set_target_velocity(target_velocity_rad);
  right_motor->update();

  double left_velocity_rad = left_motor->get_velocity();
  double right_velocity_rad = right_motor->get_velocity();
  Serial.print("left_motor:");
  Serial.print(left_velocity_rad);
  Serial.print(",");
  Serial.print("right_motor:");
  Serial.print(right_velocity_rad);
  Serial.print(",");
  Serial.print("target:");
  Serial.print(target_velocity_rad);
  Serial.println();

  delay(20);
} 
