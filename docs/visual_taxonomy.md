\# Visual Taxonomy for Pasture and Soil Conditions



This document defines the visual taxonomy used in the Pasture and Soil Vision DFence prototype.



The taxonomy is aligned with the DFence-style objective of analysing pasture and soil images collected from sources such as drone, satellite, rover, animal-mounted camera, or manual upload.



\---



\## Objective



The objective of the visual taxonomy is to define clear visual classes for pasture and soil condition monitoring.



These classes can support:



\- image annotation

\- object detection

\- semantic segmentation

\- pasture condition monitoring

\- grazing suitability estimation

\- georeferenced indicator generation

\- decision-support system integration



\---



\## Visual Classes



\### 1. Vegetation Cover



Represents visible grass, pasture, or vegetation.



Visual indicators:



\- green grass

\- dense pasture

\- continuous vegetation surface

\- healthy field cover



Use case:



\- estimate vegetation availability

\- identify potentially suitable grazing zones



\---



\### 2. Bare Soil



Represents exposed soil with little or no vegetation.



Visual indicators:



\- brown soil patches

\- dry uncovered ground

\- exposed terrain

\- low vegetation density



Use case:



\- detect reduced pasture cover

\- identify soil exposure risk



\---



\### 3. Waterlogged Soil



Represents wet, muddy, or saturated soil areas.



Visual indicators:



\- dark wet soil

\- muddy surface

\- water accumulation

\- reflective wet patches



Use case:



\- identify unsuitable grazing zones

\- support grazing avoidance decisions



\---



\### 4. Degraded Area



Represents damaged or unhealthy pasture/soil areas.



Visual indicators:



\- irregular vegetation

\- damaged pasture patches

\- sparse or unhealthy cover

\- mixed exposed soil and vegetation



Use case:



\- monitor pasture degradation

\- support recovery or intervention planning



\---



\### 5. Overgrazed Area



Represents areas affected by excessive grazing pressure.



Visual indicators:



\- very short vegetation

\- exposed soil mixed with sparse grass

\- patchy pasture

\- visible grazing pressure



Use case:



\- identify overused grazing zones

\- support dynamic virtual fence decisions



\---



\### 6. Suitable Grazing Area



Represents areas with healthy pasture conditions suitable for grazing.



Visual indicators:



\- good vegetation cover

\- low bare soil

\- no major waterlogging

\- no obvious degradation



Use case:



\- identify areas suitable for grazing

\- support decision-support recommendations



\---



\## Taxonomy Levels



The taxonomy can be organized into three levels:



```text

Level 1: Surface Type

\- vegetation

\- soil

\- wet/waterlogged surface



Level 2: Condition

\- healthy

\- degraded

\- overgrazed

\- waterlogged



Level 3: Decision Support

\- suitable for grazing

\- avoid grazing

\- needs recovery

\- requires inspection

