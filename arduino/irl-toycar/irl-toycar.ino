#include <ESP32Encoder.h> // https://github.com/madhephaestus/ESP32Encoder.git 
#include <PID_v1.h>
 
#define LEFT_ENC_A 15
#define LEFT_ENC_B 2 
#define LEFT_PWM 21
#define LEFT_DIR 22
#define LEFT_SLP 23

#define RIGHT_ENC_A 14
#define RIGHT_ENC_B 12 
#define RIGHT_PWM 32
#define RIGHT_DIR 35
#define RIGHT_SLP 34


unsigned long time_last;
unsigned long time_last_print;
long left_position, left_position_last, right_position, right_position_last;
ESP32Encoder left_encoder, right_encoder;

double velocity_left;
double velocity_right;
double pid_target, pid_left_out, pid_right_out;
double Kp=1.0, Ki=35, Kd=0.1;
PID pid_left(&velocity_left, &pid_left_out, &pid_target, Kp, Ki, Kd, DIRECT);
PID pid_right(&velocity_right, &pid_right_out, &pid_target, Kp, Ki, Kd, DIRECT);


void setup () { 
  ESP32Encoder::useInternalWeakPullResistors = puType::up;

  left_encoder.attachFullQuad ( LEFT_ENC_A, LEFT_ENC_B );
  left_encoder.setCount ( 0 );

  right_encoder.attachFullQuad ( RIGHT_ENC_A, RIGHT_ENC_B );
  right_encoder.setCount ( 0 );

  pinMode(LEFT_SLP, OUTPUT);
  pinMode(LEFT_PWM, OUTPUT);
  pinMode(LEFT_DIR, OUTPUT);

  pinMode(RIGHT_SLP, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);
  pinMode(RIGHT_DIR, OUTPUT);

  pid_target = M_PI;
  pid_left.SetMode(AUTOMATIC);
  pid_right.SetMode(AUTOMATIC);


  time_last = millis();
  time_last_print = millis();

  Serial.begin ( 115200 );
}

void update_readings() {
  time_last = millis();
  right_position_last = right_position;
  left_position_last = left_position;
}

void loop () {
  pid_target = sin(millis()/1000.0) * M_PI + M_PI;
  left_position = left_encoder.getCount();
  right_position = right_encoder.getCount();

  float delta_time_s = (millis() - time_last) / 1000.0;
  long delta_right_position = right_position - right_position_last;
  long delta_left_position = left_position - left_position_last;
  float delta_right_position_rad = (delta_right_position/1440.0) * 2 * M_PI;
  float delta_left_position_rad = (delta_left_position/1440.0) * 2 * M_PI;

  velocity_left = delta_left_position_rad/delta_time_s;
  velocity_right = delta_right_position_rad/delta_time_s;

  pid_left.Compute();
  pid_right.Compute();

  Serial.print("left_count:");
  Serial.print(velocity_left);
  // Serial.print(left_position / 1440);
  Serial.print(",");
  Serial.print("right_velocity:");
  Serial.print(velocity_right);
  // Serial.print(",");
  // Serial.print("pid_output:");
  // Serial.print(pid_out);
  Serial.print(",");
  Serial.print("pid_target:");
  Serial.print(pid_target);
  Serial.println();
  // Serial.print(right_position / 1440);
  time_last_print = millis();

  digitalWrite(LEFT_SLP, HIGH);  // read the input pin
  digitalWrite(LEFT_DIR, LOW);  // read the input pin
  analogWrite(LEFT_PWM, min(63, (int)pid_left_out)); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255

  // digitalWrite(RIGHT_SLP, HIGH);  // read the input pin
  // digitalWrite(RIGHT_DIR, LOW);  // read the input pin
  analogWrite(RIGHT_PWM, min(63, (int)pid_right_out)); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255

  update_readings();
  delay(20);
} 

// int ledPin = 16;      // LED connected to digital pin 9
// int analogPin = 17;   // potentiometer connected to analog pin 3

// void setup() {
//   Serial.begin(9600);
//   pinMode(ledPin, OUTPUT);  // sets the pin as output
//   pinMode(analogPin, OUTPUT);  // sets the pin as output
// }

// void loop() {
//   Serial.println("going");
//   digitalWrite(ledPin, HIGH);  // read the input pin
//   analogWrite(analogPin, 127); // analogRead values go from 0 to 1023, analogWrite values from 0 to 255
//   delay(50);
// }
