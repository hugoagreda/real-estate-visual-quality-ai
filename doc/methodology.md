This document describes the methodology used to evaluate and score the visual quality of real estate images using explainable computer vision techniques.

# Metric 1 — Lighting Quality Evaluation

## 1. General Description

Lighting is one of the most influential factors in the perceived quality of real estate photography.  
Poor lighting conditions often convey a sense of reduced space, neglect, or low-quality presentation, regardless of the actual condition of the property.

For this reason, lighting quality is considered a core metric within the proposed system.

---

## 2. Objective of the Metric

The objective of this metric is to:
- Assess whether an image presents adequate lighting conditions
- Detect underexposed or overexposed images
- Evaluate the distribution of light across the scene
- Provide an objective and explainable lighting quality score

This metric does **not** evaluate artistic style or aesthetic preferences, focusing solely on technical lighting quality.

---

## 3. Evaluation Criteria

The lighting quality metric is composed of three main components:

### 3.1 Global Brightness Level
The average brightness of the image is analyzed to determine whether:
- The image is underexposed
- The image is correctly exposed
- The image is overexposed

### 3.2 Light Distribution
The system evaluates whether light is:
- Evenly distributed across the image
- Concentrated in a specific region
- Producing excessive shadows in relevant areas

### 3.3 Extreme Intensity Regions
The presence of extreme regions is detected, including:
- Overexposed areas (clipped highlights)
- Underexposed areas with loss of detail

---

## 4. Technical Approach (Computer Vision)

Lighting evaluation is performed using classical computer vision techniques:

- Conversion of the image to grayscale
- Histogram analysis of pixel intensity values
- Computation of statistical measures:
  - mean brightness
  - standard deviation
  - proportion of pixels in extreme intensity ranges

This approach allows for an objective and reproducible evaluation without requiring complex model training at early stages.

---

## 5. Scoring System

The lighting metric produces a normalized score ranging from **0 to 100**, where:

- **0–30** → Poor lighting quality  
- **31–60** → Acceptable but improvable lighting  
- **61–85** → Good lighting quality  
- **86–100** → Optimal lighting quality  

The final lighting score is computed by combining:
- average brightness level
- uniformity of light distribution
- penalties for extreme intensity regions

---

## 6. Interpretability and Explanations

Explainability is a key design goal of the system.  
Alongside the numerical score, the system generates descriptive feedback such as:

- “The image presents insufficient lighting in several areas.”
- “Lighting conditions are acceptable, but strong shadows are present.”
- “Lighting is well balanced and suitable for publication.”

These explanations help users clearly understand the assessment results.

---

## 7. Recommended Improvements

When lighting quality is classified as improvable, the system may:

- Recommend increasing overall exposure
- Suggest improving light distribution
- Apply light automatic enhancements, such as:
  - brightness adjustment
  - contrast correction
  - basic shadow recovery

All automatic enhancements are **reversible** and do not alter the physical characteristics of the property.

---

## 8. Metric Limitations

This metric:
- Does not assess aesthetic appeal
- Does not consider artistic photographic intent
- Does not distinguish between natural and artificial lighting
- Does not replace professional photographic judgment

Its purpose is to provide a **technical and objective evaluation**, not a subjective critique.

---

## 9. Integration within the Overall System

The lighting quality score represents one of the components contributing to the **global visual quality score**, together with other metrics such as sharpness, composition, color balance, and visual clutter.
The relative weight of this metric within the global score is defined in later stages of the project.

# Metric 2 — Sharpness and Blur Evaluation

## 1. General Description

Image sharpness is a fundamental factor in perceived image quality.  
Blurred or out-of-focus images reduce clarity, convey unprofessionalism, and negatively impact user perception, especially in real estate listings where visual detail is essential.

This metric evaluates the level of sharpness of an image and detects the presence of blur caused by camera motion, incorrect focus, or low light conditions.

---

## 2. Objective of the Metric

The objective of this metric is to:
- Determine whether an image is sharp or blurred
- Quantify the degree of sharpness loss
- Provide an objective and explainable sharpness score
- Identify images that may require re-capture or enhancement

This metric focuses on **technical sharpness**, not artistic depth-of-field effects.

---

## 3. Evaluation Criteria

Sharpness evaluation is based on the analysis of edge and texture information within the image.

### 3.1 Edge Strength
Sharp images contain well-defined edges and transitions between objects.  
Blurred images exhibit smoother transitions and reduced edge contrast.

### 3.2 High-Frequency Detail
Sharp images preserve fine textures and details, while blur attenuates high-frequency components.

### 3.3 Global Consistency
The metric evaluates whether blur affects:
- The entire image (motion blur)
- Localized regions (focus issues)

---

## 4. Technical Approach (Computer Vision)

The sharpness metric is computed using classical computer vision techniques, including:

- Conversion of the image to grayscale
- Application of edge detection operators
- Measurement of edge variance and intensity
- Frequency-based analysis to estimate detail loss

Common techniques include:
- Laplacian variance
- Gradient magnitude analysis

These methods provide a reliable and computationally efficient estimation of image sharpness.

---

## 5. Scoring System

The sharpness metric produces a normalized score between **0 and 100**, where:

- **0–30** → Severely blurred image  
- **31–60** → Noticeable blur  
- **61–85** → Acceptable sharpness  
- **86–100** → High sharpness  

The final sharpness score is derived from:
- edge strength measurements
- variance of high-frequency components
- penalties for uniform smoothness

---

## 6. Interpretability and Explanations

To ensure transparency, the system generates interpretable feedback such as:

- “The image appears blurred and lacks sharp edges.”
- “Moderate blur detected, possibly due to camera movement.”
- “The image is sharp and well-focused.”

These explanations allow users to understand whether quality issues are due to capture conditions rather than content.

---

## 7. Recommended Improvements

When sharpness is classified as insufficient, the system may suggest:

- Retaking the photograph with better stabilization
- Improving lighting conditions to reduce motion blur
- Applying light sharpening filters (if applicable)

Automatic sharpening is applied cautiously to avoid introducing artifacts or unrealistic textures.

---

## 8. Metric Limitations

This metric:
- Does not evaluate artistic depth-of-field
- Cannot distinguish intentional blur from accidental blur
- May be affected by low-resolution images
- Does not replace professional photographic assessment

The goal is to provide a **technical quality indicator**, not an artistic judgment.

---

## 9. Integration within the Overall System

The sharpness score contributes to the **global visual quality score** alongside other metrics such as lighting, composition, color balance, and visual clutter.
Its relative weight within the global score is defined during system calibration and evaluation.

# Metric 3 — Composition and Geometric Alignment Evaluation

## 1. General Description

Image composition plays a crucial role in how interior spaces are perceived.  
In real estate photography, poor composition — such as tilted vertical lines, unbalanced framing, or incorrect perspective — can distort the perception of space and reduce the professional quality of an image.

This metric focuses on evaluating the **geometric and structural correctness** of the image composition rather than aesthetic or artistic choices.

---

## 2. Objective of the Metric

The objective of this metric is to:
- Assess whether the image is properly aligned and well framed
- Detect geometric distortions caused by camera tilt or perspective issues
- Identify composition problems that negatively affect spatial perception
- Provide clear and actionable feedback to improve image alignment

This metric does **not** evaluate artistic composition styles or decorative preferences.

---

## 3. Evaluation Criteria

The composition and alignment metric is based on three main aspects:

### 3.1 Vertical Line Alignment
Interior environments contain strong vertical structures such as walls, doors, and windows.  
This component evaluates whether vertical lines appear straight or tilted.
Misaligned verticals often result from incorrect camera tilt and significantly reduce perceived image quality.

---

### 3.2 Horizon and Camera Tilt
The system evaluates whether the image is:
- Horizontally level
- Tilted to the left or right

Even small deviations can create a sense of imbalance and amateur photography.

---

### 3.3 Framing Balance
This component analyzes whether the main visual content is:
- Properly centered
- Excessively cropped
- Unbalanced toward one side of the image

Balanced framing improves spatial readability and visual comfort.

---

## 4. Technical Approach (Computer Vision)

The composition metric is implemented using classical geometric analysis techniques:

- Edge detection to identify dominant structural lines
- Line detection methods (e.g., Hough Transform)
- Estimation of dominant vertical and horizontal orientations
- Measurement of angular deviations from ideal alignment

These methods allow objective evaluation of geometric correctness without requiring semantic understanding of the scene.

---

## 5. Scoring System

The composition metric outputs a normalized score between **0 and 100**, where:

- **0–30** → Poor composition and strong misalignment  
- **31–60** → Noticeable alignment issues  
- **61–85** → Acceptable composition  
- **86–100** → Well-aligned and balanced composition  

The final composition score is computed by combining:
- vertical alignment accuracy
- horizontal leveling accuracy
- framing balance consistency

---

## 6. Interpretability and Explanations

To maintain system transparency, the following types of explanations may be generated:

- “Vertical lines appear tilted, indicating camera misalignment.”
- “The image is slightly rotated and could benefit from correction.”
- “The image is well aligned and geometrically balanced.”

These messages allow users to easily understand the detected issues and their impact.

---

## 7. Recommended Improvements

When composition issues are detected, the system may recommend or apply:

- Automatic rotation to correct camera tilt
- Vertical perspective correction
- Cropping adjustments to improve framing balance

All automatic corrections are conservative and reversible to preserve realism.

---

## 8. Metric Limitations

This metric:
- Does not evaluate artistic composition styles
- Does not consider furniture layout or decoration
- May be affected by scenes with few detectable straight lines
- Does not assess aesthetic balance beyond geometric structure

Its purpose is to evaluate **technical alignment and framing quality**, not visual taste.

---

## 9. Integration within the Overall System

The composition score contributes to the **global visual quality score** together with lighting, sharpness, color balance, and visual clutter metrics.
Its weighting within the global score is defined during system calibration and evaluation phases.

# Metric 4A — Color Balance and Saturation Evaluation

## 1. General Description

Color balance and saturation strongly influence the perceived realism and quality of real estate images.  
Incorrect color balance can introduce unnatural color casts (e.g., yellowish or bluish tones), while inappropriate saturation levels may make images appear dull or artificially enhanced.

This metric evaluates the technical correctness of color representation, focusing on realism and visual comfort rather than artistic styling.

---

## 2. Objective of the Metric

The objective of this metric is to:
- Detect color imbalances and dominant color casts
- Evaluate whether color saturation levels are appropriate
- Identify images with unnatural or distorted color representation
- Provide objective and explainable feedback regarding color quality

This metric does **not** assess artistic color grading or stylistic choices.

---

## 3. Evaluation Criteria

Color quality evaluation is based on two main components:

### 3.1 Color Balance
This component analyzes whether the image presents a dominant color cast caused by lighting conditions or camera settings.

Typical issues include:
- Excessive warm tones (yellow/orange dominance)
- Excessive cool tones (blue dominance)
- Green or magenta color shifts

A well-balanced image should exhibit neutral whites and consistent color distribution.

---

### 3.2 Color Saturation
This component evaluates whether colors are:
- Under-saturated (flat or lifeless appearance)
- Over-saturated (unnatural or exaggerated colors)
- Within an acceptable saturation range

Proper saturation contributes to a realistic and appealing representation of the space.

---

## 4. Technical Approach (Computer Vision)

Color balance and saturation are evaluated using color space analysis:

- Conversion of the image from RGB to alternative color spaces (e.g., HSV or LAB)
- Analysis of chromatic channels to detect dominant color shifts
- Measurement of average and variance of saturation values
- Detection of abnormal saturation distributions

This approach enables robust color analysis independent of image brightness.

---

## 5. Scoring System

The color balance and saturation metric outputs a normalized score between **0 and 100**, where:

- **0–30** → Poor color balance and saturation  
- **31–60** → Noticeable color issues  
- **61–85** → Acceptable and natural color representation  
- **86–100** → Well-balanced and realistic colors  

The final color score is computed by combining:
- color balance deviation penalties
- saturation range evaluation
- penalties for extreme color distortions

---

## 6. Interpretability and Explanations

To ensure transparency, the system provides interpretable explanations such as:

- “The image shows a strong warm color cast.”
- “Color saturation appears excessive and unnatural.”
- “Colors are well balanced and visually realistic.”

These messages help users understand the source of visual quality issues.

---

## 7. Recommended Improvements

When color issues are detected, the system may recommend or apply:

- Automatic white balance correction
- Saturation adjustment within safe limits
- Minor color normalization to restore natural appearance

All color corrections are conservative and reversible, preserving the authenticity of the property.

---

## 8. Metric Limitations

This metric:
- Does not evaluate artistic color grading
- Does not account for intentional stylistic filters
- May be affected by mixed lighting conditions
- Does not assess material or surface quality

Its purpose is to ensure **technical color fidelity**, not artistic expression.

---

## 9. Integration within the Overall System

The color balance and saturation score contributes to the **global visual quality score**, alongside lighting, sharpness, composition, and visual clutter metrics.
Its weighting within the global score is defined during system calibration and evaluation.

# Metric 4B — Visual Clutter and Scene Complexity Evaluation

## 1. General Description

Visual clutter refers to the presence of excessive or disorganized visual elements within an image, which can negatively impact clarity, spatial perception, and overall image quality.  
In real estate photography, high visual clutter often makes spaces appear smaller, less organized, and less appealing, regardless of their actual size or condition.

This metric evaluates the **perceived visual complexity** of an image, focusing on how easily the space can be visually understood.

---

## 2. Objective of the Metric

The objective of this metric is to:
- Measure the level of visual clutter within an image
- Identify images with excessive visual complexity
- Penalize scenes that reduce spatial readability
- Provide objective and explainable feedback related to visual organization

This metric does **not** evaluate cleanliness, decoration quality, or lifestyle choices.

---

## 3. Evaluation Criteria

Visual clutter evaluation is based on the analysis of structural and textural complexity within the image.

### 3.1 Edge Density
Images with a high number of edges and abrupt transitions tend to appear visually cluttered.

This component measures:
- The density of detected edges
- The distribution of edge information across the image

High edge density often correlates with excessive visual noise.

---

### 3.2 Texture Complexity
This component evaluates the amount of fine-grained texture present in the scene.

Highly textured images with many small details may:
- Reduce visual clarity
- Distract attention from the overall space

Texture complexity is analyzed using statistical texture descriptors.

---

### 3.3 Object Distribution Uniformity
The metric evaluates whether visual elements are:
- Evenly distributed
- Highly concentrated in specific regions

Strong concentration of elements in limited areas increases perceived clutter.

---

## 4. Technical Approach (Computer Vision)

Visual clutter is assessed using low-level image analysis techniques:

- Edge detection to estimate edge density
- Texture analysis using local descriptors
- Spatial distribution analysis of visual features
- Entropy-based measures to quantify scene complexity

These techniques allow clutter estimation without explicit object recognition.

---

## 5. Scoring System

The visual clutter metric produces a normalized score between **0 and 100**, where:

- **0–30** → High visual clutter  
- **31–60** → Moderate visual clutter  
- **61–85** → Low visual clutter  
- **86–100** → Very clean and visually simple scene  

The final clutter score is computed by combining:
- edge density penalties
- texture complexity penalties
- spatial concentration penalties

Higher scores indicate lower perceived clutter.

---

## 6. Interpretability and Explanations

To maintain transparency, the system generates explanations such as:

- “The image contains a high density of visual elements, reducing clarity.”
- “Visual complexity is moderate and may benefit from decluttering.”
- “The scene is visually clean and easy to interpret.”

These explanations help users understand how visual clutter affects image quality.

---

## 7. Recommended Improvements

When high visual clutter is detected, the system may recommend:

- Reducing the number of visible objects
- Removing distracting elements from the foreground
- Improving spatial organization before capturing the image

No automatic object removal is performed to avoid altering the scene.

---

## 8. Metric Limitations

This metric:
- Does not identify specific objects
- Does not judge interior design quality
- Cannot distinguish intentional decorative richness from clutter
- May be influenced by architectural complexity

Its goal is to estimate **perceived visual complexity**, not stylistic value.

---

## 9. Integration within the Overall System

The visual clutter score contributes to the **global visual quality score**, complementing lighting, sharpness, composition, and color balance metrics.
This metric plays a key role in assessing spatial readability and perceived cleanliness of real estate images.

# Global Visual Quality Scoring Strategy

## 1. Overview

The global visual quality score represents a unified and interpretable evaluation of a real estate image based on multiple technical visual metrics.  
Rather than relying on a single criterion, the system combines complementary metrics to reflect overall perceived image quality in a robust and explainable manner.

The final score is designed to be:
- Objective
- Reproducible
- Easy to interpret
- Suitable for comparison between images

---

## 2. Contributing Metrics

The global score is computed using the following five metrics:

1. Lighting Quality  
2. Sharpness and Blur  
3. Composition and Geometric Alignment  
4. Color Balance and Saturation  
5. Visual Clutter and Scene Complexity  

Each metric produces an independent normalized score ranging from **0 to 100**.

---

## 3. Weighting Strategy

To reflect the relative importance of each visual aspect in real estate photography, a weighted aggregation strategy is applied.

The initial weighting scheme for the first version of the system (V1) is defined as follows:

| Metric | Weight |
|------|--------|
| Lighting | 25% |
| Sharpness | 20% |
| Composition | 20% |
| Color Balance & Saturation | 15% |
| Visual Clutter | 20% |

Lighting is assigned the highest weight due to its dominant influence on perceived image quality, while the remaining metrics contribute proportionally to the overall assessment.

Weights may be adjusted in future iterations based on empirical evaluation and user feedback.

---

## 4. Score Normalization

All metric scores are normalized to a common scale between **0 and 100** prior to aggregation.  
This ensures comparability and prevents any single metric from dominating the global score due to scale differences.

Normalization allows the system to remain modular and extensible.

---

## 5. Global Score Computation

The global visual quality score is computed as a weighted sum of the individual metric scores:
Global Score = Σ (metric_score × metric_weight)

The resulting score is then rounded to the nearest integer for presentation purposes.

---

## 6. Global Score Interpretation

The final global score is interpreted using qualitative categories to improve user understanding:

- **0–30** → Poor visual quality  
- **31–60** → Acceptable but improvable quality  
- **61–80** → Good visual quality  
- **81–100** → High visual quality  

These categories allow users to quickly assess whether an image is suitable for publication or requires improvement.

---

## 7. Explainability and Transparency

In addition to the numerical global score, the system provides:
- A breakdown of individual metric scores
- Explanatory messages for each metric
- Clear indications of the most influential factors affecting the final score

This approach ensures transparency and prevents the system from behaving as a black box.

---

## 8. Handling Edge Cases

The scoring strategy accounts for potential edge cases, such as:
- Extremely poor performance in a single critical metric
- Images with mixed-quality characteristics

In such cases, explanatory feedback highlights the dominant issues affecting the global score.

---

## 9. Future Improvements

Future iterations of the scoring strategy may include:
- Dynamic weight adjustment based on user feedback
- Context-aware weighting depending on room type
- Data-driven optimization of weights using supervised learning

These enhancements are considered beyond the scope of the initial project version.