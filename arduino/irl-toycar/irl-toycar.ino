#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <Encoder.h>
#include <stdexcept>
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

#define SERVICE_UUID "a74f372c-6a91-4638-85a8-9362f691f964"
#define COMMAND_CHARACTERISTIC_UUID "eee4879a-80a8-4d33-af74-b05826ee658f"

double pid_k_proportional = 1.0;
double pid_k_integral = 35;
double pid_k_derivative = 0.1;
long encoder_cpr = 1440;
double max_velocity_rad = 4 * M_PI;

Motor* left_motor;
Motor* right_motor;
Encoder* left_encoder;
Encoder* right_encoder;

BLEServer* ble_server;
BLEService* ble_service;
BLECharacteristic* command_characteristic;
unsigned long last_request_millis;
unsigned long timeout_millis = 1000;


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

  BLEDevice::init("toycar");
  ble_server = BLEDevice::createServer();
  ble_service = ble_server->createService(SERVICE_UUID);
  command_characteristic = ble_service->createCharacteristic(COMMAND_CHARACTERISTIC_UUID, BLECharacteristic::PROPERTY_WRITE_NR);
  command_characteristic->setValue("0.0 0.0");
  ble_service->start();
  BLEAdvertising *ble_advertising = BLEDevice::getAdvertising();
  ble_advertising->addServiceUUID(SERVICE_UUID);
  ble_advertising->setScanResponse(true);
  ble_advertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
  ble_advertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
}

void loop () {
  double left_target_velocity_rad = 0.0;
  double right_target_velocity_rad = 0.0;
  try {
    std::string command = command_characteristic->getValue().c_str();
    size_t next_index;
    left_target_velocity_rad = std::stod(command, &next_index);
    right_target_velocity_rad = std::stod(command.substr(next_index));
    left_motor->set_target_velocity(left_target_velocity_rad);
    right_motor->set_target_velocity(right_target_velocity_rad);
  } catch(const std::invalid_argument& err) {
    Serial.printf("Invalid argument: %s, skipping. \n", err.what());
  }

  left_motor->update();
  right_motor->update();

  double left_velocity_rad = left_motor->get_velocity();
  double right_velocity_rad = right_motor->get_velocity();
  Serial.print("left_motor:");
  Serial.print(left_velocity_rad);
  Serial.print(",");
  Serial.print("right_motor:");
  Serial.print(right_velocity_rad);
  Serial.print(",");
  Serial.print("left_target:");
  Serial.print(left_target_velocity_rad);
  Serial.print(",");
  Serial.print("right_target:");
  Serial.print(right_target_velocity_rad);
  Serial.println();

  delay(20);
} 
