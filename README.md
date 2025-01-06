### Computational Biology with Mechanistic Modeling for the Prediction of Antimicrobial Resistance in Novel Antibiotics



####  **Description**

This project explores the prediction of **antimicrobial resistance (AMR)** using a combination of computational models, including **machine learning models**, **mechanistic modeling**, and **agent-based modeling (ABM)**. AMR poses a significant challenge in public health, as resistant bacteria render many antibiotics ineffective. Our study focuses on identifying resistance patterns across **four key antibiotics**: **Ciprofloxacin (CIP), Cefotaxime (CTX), Ceftazidime (CTZ), and Gentamicin (GEN)**. 

The project leverages both **traditional machine learning models** (Random Forest and MLP Neural Network) and **mechanistic and agent-based approaches** to simulate the dynamics of resistance evolution, offering deep biological insights into AMR. 



#### **Dataset Overview**

The dataset contains **antibiotic resistance phenotypes** for a collection of bacterial samples. Each sample is classified as either resistant or non-resistant for each of the four antibiotics.  

**Dataset Summary:**
- **Columns:**  
   - `CIP`: Resistance to Ciprofloxacin  
   - `CTX`: Resistance to Cefotaxime  
   - `CTZ`: Resistance to Ceftazidime  
   - `GEN`: Resistance to Gentamicin  
   - Other identifiers and metadata for bacterial samples.

**Key Insights from the Dataset:**
- Resistance patterns across antibiotics suggest potential **cross-resistance**, where resistance to one antibiotic correlates with resistance to others.
- Some antibiotics have **imbalanced resistance counts**, which may reflect real-world trends in antibiotic usage and selective pressure.
- Identifying the most influential features allows us to investigate potential **biological markers of resistance**, aligning with our mechanistic modeling goals.



#### **Methodology**

1. **Data Preprocessing**  
   - The dataset was cleaned and split into **train and test sets** using an 80/20 ratio.
   - Standardization was applied to features for MLP Neural Network training.

2. **Machine Learning Models**  
   - **Random Forest Classifier**: Known for its interpretability and ability to capture feature importance, helping us identify potential resistance markers.
   - **MLP Neural Network**: Used to model **non-linear relationships**, which might reflect complex biological gene interactions affecting resistance.

3. **Mechanistic Modeling**  
   - A **logistic growth model** simulates bacterial population dynamics over time, based on the resistance data. The carrying capacity and growth rate parameters reflect the observed frequency of resistance in the dataset.

4. **Agent-Based Modeling (ABM)**  
   - ABM simulates the **evolution of resistance** at the individual bacterial level, incorporating randomness to reflect the stochastic nature of resistance gain or loss.

5. **Model Validation and Performance Evaluation**  
   - **Confusion matrices**, **ROC curves**, and **AUC scores** provide detailed evaluations of model performance.  
   - **Hyperparameter tuning** was applied to Random Forest for optimal performance.



#### **Computational Biological Insights**

1. **Random Forest Model:**
   - Achieved **Accuracy = 1.00** and **AUC = 1.00**, indicating it can perfectly classify resistant and non-resistant strains in the current dataset.
   - Important features identified (CIP, CTX, CTZ, and GEN) suggest potential **biological markers for resistance**.
   - **Interpretability** of this model allows deeper exploration of how feature interactions impact resistance.

2. **MLP Neural Network:**
   - Also achieved **Accuracy = 1.00** and **AUC = 1.00**, capturing **non-linear patterns** that may reflect gene interactions or environmental factors affecting resistance.
   - While harder to interpret, MLP models provide insights into the **complexity** of AMR beyond linear relationships.

3. **Mechanistic Modeling:**
   - **Logistic growth simulations** align with observed resistance dynamics, showing how bacterial populations stabilize under environmental constraints.
   - The growth curve models potential changes in population size under antibiotic pressure, aligning with trends in AMR datasets.

4. **Agent-Based Modeling:**
   - ABM captures **dynamic resistance shifts**, showing how individual bacteria might evolve resistance traits over time. 
   - Provides a **stochastic simulation** of resistance development, complementing the deterministic approach of mechanistic models.

5. **Cross-Resistance Trends:**
   - Both models highlight potential **cross-resistance**, a phenomenon where resistance to one antibiotic leads to resistance to others. This finding aligns with the biological behavior of multi-drug resistant bacterial strains.


#### **Visualizations and Results**

1. **Class Distribution of Antibiotic Resistance**  
   - Bar plot showing the number of resistant samples for each antibiotic.

2. **Confusion Matrices for Random Forest and MLP Models**  
   - Visualize model predictions against actual values, providing insights into precision and recall.

3. **ROC Curves for Both Models**  
   - Evaluate the ability of each model to distinguish between resistant and non-resistant strains.

4. **Model Comparison Plot**  
   - Compare the accuracy and AUC scores of both models to determine the better-performing model.

5. **Mechanistic Modeling Plot**  
   - Logistic growth curve visualizing bacterial population dynamics over time.

6. **Agent-Based Modeling Plot**  
   - Evolution of resistance over time, demonstrating how resistance traits develop in a simulated bacterial population.


#### **Conclusion and Future Work**

This project demonstrates the power of **computational biology** in understanding and predicting AMR trends using both machine learning and mechanistic modeling approaches. The following outcomes were achieved:  
1. **High-performance models** (Random Forest and MLP) capable of predicting antibiotic resistance with perfect accuracy on the given dataset.
2. **Mechanistic and ABM simulations** aligned with the dataset, providing insights into resistance dynamics and cross-resistance behavior.
3. **Interpretability of Random Forest** allows deeper biological exploration, while MLP captures **non-linear patterns**.

**Future Work:**  
- Explore **gene-level resistance markers** using advanced biological analyses.
- Test models on larger, **real-world datasets** for further validation.
- Extend ABM to incorporate **mutation events** and **horizontal gene transfer**, better reflecting real-world AMR dynamics.
