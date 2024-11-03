#include <ESP32Encoder.h> // https://github.com/madhephaestus/ESP32Encoder.git 
#include "motor.h"
 
#define LEFT_ENC_A 15
#define LEFT_ENC_B 2 
#define LEFT_PWM 21
#define LEFT_DIR 22
#define LEFT_SLP 23

#define RIGHT_ENC_A 14
#define RIGHT_ENC_B 12 
#define RIGHT_PWM 27
#define RIGHT_DIR 26
#define RIGHT_SLP 25

double pid_k_proportional = 1.0;
double pid_k_integral = 35;
double pid_k_derivative = 0.1;
long encoder_cpr = 1440;
double max_velocity_rad = 2 * M_PI;

Motor* left_motor;
Motor* right_motor;
ESP32Encoder left_encoder, right_encoder;




void setup () { 
  ESP32Encoder::useInternalWeakPullResistors = puType::up;
  left_encoder.attachFullQuad(LEFT_ENC_A, LEFT_ENC_B);
  left_encoder.setCount(0);
  right_encoder.attachFullQuad(RIGHT_ENC_A, RIGHT_ENC_B);
  right_encoder.setCount(0);

  
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
