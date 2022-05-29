#include<cvzone.h>

SerialData serialData(3, 1);

int red = 8;
int green = 9;
int blue = 10;
int valsRec[3];

void setup() {
  // put your setup code here, to run once:
  serialData.begin();
  pinMode(red, OUTPUT);
  pinMode(green, OUTPUT);
  pinMode(blue, OUTPUT);

}

void loop() {
  // put your main code here, to run repeatedly:
  serialData.Get(valsRec);
  digitalWrite(red, valsRec[0]);
  digitalWrite(green, valsRec[1]);
  digitalWrite(blue, valsRec[2]);

}
