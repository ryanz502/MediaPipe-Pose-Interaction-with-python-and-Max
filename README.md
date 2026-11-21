# MediaPipe-Pose-Interaction-with-python-and-Max

This project is an interactive body-tracking system developed using Python, MediaPipe, and Max/MSP. It is designed for live performance, with real-time mapping calibration and backup MIDI control in case of tracking failure.


## 1. System Overview

The system consists of a Python script and a Max/MSP patch.

The Python script also calculates the three-dimensional velocity and acceleration of both wrists based on their positional changes over time. These data are transmitted separately from the landmark coordinates via OSC.

The Python script must be launched before the Max patch. It uses **MediaPipe Pose** to track six body landmarks:

- Left and right shoulders
- Left and right elbows
- Left and right wrists

The landmark data is transmitted to Max/MSP for monitoring and mapping.

Currently, the Max patch only maps the coordinates of the two wrists. It does not process audio directly, but provides tools for monitoring and adjusting mapping parameters and switching between body-tracking and backup MIDI control.

## 2. Setup

Before running the system, configure the destination IP address and port in the `udpsend` object.

Launch the Python script first, then open the Max patch to receive the tracking data.

## 3. Real-Time Mapping Calibration

The mapping ranges can be adjusted during performance rehearsals.

1. Click the **blue `toggle`** object to activate calibration.
2. Ask the performer to move within the camera's tracking area while adjusting the mapping ranges.
3. Once a suitable range has been established, click **Apply** to confirm the settings.
4. Turn off the blue `toggle` to stop calibration.

This allows the mapping to be adapted to different performers and performance spaces.

## 4. Backup MIDI Control

The Max patch includes a fallback mechanism designed to maintain control during live performances.

If tracking data stops arriving due to a camera failure, Python script interruption, or another connection issue, the system automatically switches to **MIDI control after five seconds without incoming data**.

To configure the backup MIDI input, locate the **purple `button` and `umenu` objects** in the upper-right corner of the Max patch:

1. Click the purple `button` to retrieve available MIDI device information.
2. Use the purple `umenu` to select an external backup MIDI device.

Possible backup devices include wearable sensors with their accompanying software or a conventional MIDI controller.

If no external MIDI device is available, the two `pictslider` objects can also be used to adjust the control values manually with a mouse.

**Note:** The backup MIDI and manual control functions are intended as emergency alternatives rather than the primary interaction method.

## 5. Intended Application

This system is designed for interactive music and live performance projects involving body movement and gestural control. Its real-time calibration and fallback mechanisms aim to improve adaptability and reliability during rehearsals and performances.
