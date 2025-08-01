#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "model_xgb_ph.h"
#include "model_xgb_tur.h"
#include "model_xgb_cond.h"

// OLED display settings
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Button pins
const int btnNext = 12; // Next/Select
const int btnOK = 14;   // Save/OK
const int btnBack = 27; // Back/Reset

// Sensor analog pins
const int pinPH = A0;
const int pinTUR = A1;
const int pinCOND = A2;

float ph = 0, tur = 0, cond = 0;
float ph_next = 0, tur_next = 0, cond_next = 0;

enum MenuState
{
    MENU_START,
    MENU_PH,
    MENU_TUR,
    MENU_COND,
    MENU_PREDICT,
    MENU_RESULT
};
MenuState menu = MENU_START;

// Prototypes
float readPHSensor();
float readTurbiditySensor();
float readTDSSensor();
void updateDisplay();

void setup()
{
    Serial.begin(115200);
    pinMode(btnNext, INPUT_PULLUP);
    pinMode(btnOK, INPUT_PULLUP);
    pinMode(btnBack, INPUT_PULLUP);
    pinMode(pinPH, INPUT);
    pinMode(pinTUR, INPUT);
    pinMode(pinCOND, INPUT);

    // OLED init
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
    {
        Serial.println(F("SSD1306 allocation failed"));
        for (;;)
            ;
    }
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);
    display.println("Water Quality Predictor");
    display.display();
    delay(1500);
}

void loop()
{
    static bool phSaved = false, turSaved = false, condSaved = false;

    updateDisplay(); // Update OLED content based on menu state

    switch (menu)
    {
    case MENU_START:
        if (digitalRead(btnNext) == LOW)
        {
            menu = MENU_PH;
            delay(300);
        }
        break;

    case MENU_PH:
        ph = readPHSensor();
        if (digitalRead(btnOK) == LOW)
        {
            phSaved = true;
            menu = MENU_TUR;
            delay(300);
        }
        break;

    case MENU_TUR:
        tur = readTurbiditySensor();
        if (digitalRead(btnOK) == LOW)
        {
            turSaved = true;
            menu = MENU_COND;
            delay(300);
        }
        break;

    case MENU_COND:
        cond = readTDSSensor();
        if (digitalRead(btnOK) == LOW)
        {
            condSaved = true;
            menu = MENU_PREDICT;
            delay(300);
        }
        break;

    case MENU_PREDICT:
        if (digitalRead(btnOK) == LOW)
        {
            float features[3] = {ph, tur, cond};
            ph_next = score_ph(features);
            tur_next = score_tur(features);
            cond_next = score_cond(features);
            menu = MENU_RESULT;
            delay(300);
        }
        if (digitalRead(btnBack) == LOW)
        {
            menu = MENU_START;
            phSaved = turSaved = condSaved = false;
            delay(300);
        }
        break;

    case MENU_RESULT:
        if (digitalRead(btnBack) == LOW)
        {
            menu = MENU_START;
            phSaved = turSaved = condSaved = false;
            delay(300);
        }
        break;
    }
}

// ----------- Display Handler ---------------
void updateDisplay()
{
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 0);

    switch (menu)
    {
    case MENU_START:
        display.println("Water Quality Predictor");
        display.println("Press Next to begin");
        break;
    case MENU_PH:
        display.println("Measure pH");
        display.println("Press OK to save");
        display.setTextSize(2);
        display.setCursor(0, 30);
        display.print("pH: ");
        display.print(ph, 2);
        break;
    case MENU_TUR:
        display.println("Measure Turbidity");
        display.println("Press OK to save");
        display.setTextSize(2);
        display.setCursor(0, 30);
        display.print("Tur: ");
        display.print(tur, 2);
        break;
    case MENU_COND:
        display.println("Measure Conductivity");
        display.println("Press OK to save");
        display.setTextSize(2);
        display.setCursor(0, 30);
        display.print("Cond: ");
        display.print(cond, 2);
        break;
    case MENU_PREDICT:
        display.setTextSize(1);
        display.println("Ready to Predict?");
        display.println("OK=Predict, Back=Restart");
        break;
    case MENU_RESULT:
        display.setTextSize(1);
        display.println("Predictions:");
        display.print("pH:   ");
        display.println(ph_next, 2);
        display.print("Tur:  ");
        display.println(tur_next, 2);
        display.print("Cond: ");
        display.println(cond_next, 2);
        display.println();
        display.println("Back=Restart");
        break;
    }
    display.display();
}

// ----------- Dummy Sensor Reading Functions -------------
float readPHSensor()
{
    // Replace with real sensor reading and calibration!
    return analogRead(pinPH) * (14.0 / 4095.0);
}
float readTurbiditySensor()
{
    // Replace with real sensor reading and calibration!
    return analogRead(pinTUR) * (100.0 / 4095.0);
}
float readTDSSensor()
{
    // Replace with real sensor reading and calibration!
    return analogRead(pinCOND) * (1000.0 / 4095.0);
}

// Don't forget:
// - Update sensor scaling and pins per your hardware.
// - Rename the score functions in your model .h files as score_ph, score_tur, score_cond.
