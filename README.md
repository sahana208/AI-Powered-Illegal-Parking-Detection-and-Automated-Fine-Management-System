# AI-Powered-Illegal-Parking-Detection-and-Automated-Fine-Management-System

## How the Project Works

The Illegal Parking Detection System uses computer vision and AI-based object detection to identify vehicles, recognize parking signs, determine parking status, extract license plate information, and generate an email notification for illegal parking.

### Working Flow


Input Image
     ↓
Vehicle Detection using YOLOv8
     ↓
Parking Sign Detection
     ↓
Parking Sign Classification
     ↓
Legal / Illegal Parking Decision
     ↓
License Plate Detection & OCR
     ↓
Display Result
     ↓
Email Notification for Illegal Parking


### Step 1: Upload Image

The user uploads an image containing a vehicle through the React web application.

### Step 2: Vehicle Detection

YOLOv8 detects vehicles present in the image, including:

* Car
* Motorcycle
* Bus
* Truck

### Step 3: Parking Sign Detection

The trained YOLOv8 parking-sign model detects the parking sign present in the image.

The model contains the following classes:

| Class                 | Meaning                             |
| --------------------- | ----------------------------------- |
| Parking               | Parking allowed                     |
| Parking Obstructed    | Parking sign with obstruction       |
| No Parking            | Parking prohibited                  |
| No Parking Obstructed | Parking prohibited with obstruction |
| Fake Sign             | Fake/invalid parking sign           |

### Step 4: Legal / Illegal Parking Detection

The detected parking sign is used to determine the parking status.


Parking / Parking Obstructed
            ↓
       LEGAL PARKING

No Parking / No Parking Obstructed
            ↓
      ILLEGAL PARKING


### Step 5: License Plate OCR

For an illegally parked vehicle, the system attempts to extract the license plate number using:

* OpenCV image preprocessing
* EasyOCR
* License plate pattern processing

### Step 6: Violation Notification

When an illegal parking violation is detected, the system displays the violation information and can send an email notification using SMTP.



# Results

The developed system was tested using vehicle images containing different parking conditions and parking signs.

## Result 1: Legal Parking

**Input:** Image containing a vehicle with a valid parking sign.

**System Output:**

```text
Vehicle Detected: Car
Parking Sign: Parking
Parking Status: LEGAL
```

The system correctly identifies the vehicle and recognizes that parking is allowed.

**Output Screenshot:**

Add your screenshot here:

```text
![Legal Parking Result](images/legal-parking-result.png)
```

---

## Result 2: Illegal Parking

**Input:** Image containing a vehicle in a No Parking area.

**System Output:**

```text
Vehicle Detected: Car
Parking Sign: No Parking
Parking Status: ILLEGAL
```

The system identifies the vehicle and determines that it is illegally parked.

**Output Screenshot:**

```text
![Illegal Parking Result](images/illegal-parking-result.png)
```

---

## Result 3: License Plate Recognition

For an illegally parked vehicle, the system attempts to extract the license plate using EasyOCR.

**Example Output:**

```text
Vehicle: Car
Parking Status: ILLEGAL
License Plate: [Detected Plate Number]
```

**Output Screenshot:**

```text
![License Plate OCR Result](images/license-plate-result.png)
```

---

## Result 4: Email Notification

When illegal parking is detected, the system can generate an email notification containing the violation information.

**Example:**

```text
Subject: Illegal Parking Detected

Vehicle Type: Car
Parking Status: Illegal
License Plate: [Detected Plate Number]
Detection Time: [Timestamp]
```

**Email Screenshot:**

```text
![Email Notification](images/email-notification.png)
```

---

# Overall System Output

The final system provides:

| Feature                              | Result |
| ------------------------------------ | ------ |
| Vehicle Detection                    | ✓      |
| Parking Sign Detection               | ✓      |
| Legal/Illegal Parking Classification | ✓      |
| License Plate OCR                    | ✓      |
| Violation Information                | ✓      |
| Email Notification                   | ✓      |

The system provides an end-to-end pipeline for detecting and reporting illegal parking using computer vision and deep-learning-based object detection.
