
import streamlit as st
from streamlit_option_menu import option_menu         
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.figure_factory as ff




model = joblib.load('c:/your_path/logit_model.plk')



################ Page configuration ##################################################################################################

st.set_page_config(
    page_title="Scoring Credit App", 
    page_icon="💳",  
    layout="centered",  
    initial_sidebar_state="auto",
)

# UX
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Model Explanation", "Contact"],
        icons=["credit-card", "graph-up", "envelope"],
        menu_icon="cast",
        default_index=0,
    )


################ Home ##################################################################################################

if selected == "Home":

    st.title("💳 Credit Risk Prediction")
    st.write("This app predicts the probability of a customer's non-payment based on several factors")


    # inputs users
    accounts = st.selectbox("accounts", ["", "No account", "CC < 0 euros", "CC [0-200 euros]", "CC > 200 euros"], index=0)
    history_credit = st.selectbox("history_credit", ["", "No credits or ongoing without delay", "Credits in arrears", "Past credits without delay"], index=0)
    duration_credit_value = st.slider("Duration Credit (months)", min_value=6, max_value=72, step=1)
    age_value = st.number_input("Age", min_value=18, max_value=75, step=1)
    savings = st.selectbox("savings", ["", "No savings", "> 500 euros", "< 500 euros", "No savings", "> 500 euros"], index=0)
    guarantees = st.selectbox("guarantees", ["", "With guarantor", "Without guarantor"], index=0)
    other_credits = st.selectbox("other_credits", ["", "No external credit", "External credits"], index=0)


    def discretize_duration_credit(value):
        if value < 15:
            return "[0.0, 15.0)"
        elif value <= 36:
            return "[15.0, 36.0)"
        else:
            return "[36.0, inf)"

    def discretize_age(value):
        if value < 25:
            return "[0.0, 25.0)"
        else:
            return "[25.0, inf)"

    duration_credit = discretize_duration_credit(duration_credit_value)
    age = discretize_age(age_value)


    # The inputs are stocked in a dataframe 
    input_data = pd.DataFrame({
        'accounts': [accounts],
        'history_credit': [history_credit],
        'duration_credit': [duration_credit],
        'age': [age],
        'savings': [savings],
        'guarantees': [guarantees],
        'other_credits': [other_credits]
    })


    # the prediction is ready
    if st.button("Predict"):
        if "" in [accounts, history_credit, savings, guarantees, other_credits]:
            st.error("Please fill in all the options before predicting.")
        else:
            # Continue with prediction
            prediction = model.predict(input_data)
            st.write(f"The probability of non-payment is {prediction[0]:.2%}")

            if prediction[0] > 0.5:
                st.error("Warning: Customers are at high risk of non-payment.")
            else:
                st.success("Good: The customer is at low risk of non-payment.")



####################### Data Preparation ##################################################################################################

# Loading data table and data transformation
credit2 = pd.read_csv('c:/Users/DL/Documents/MEGA/100 cas de PYTHON/Data_science_from_scratch/modelisation predictive version Python/credit2.csv') 

credit2['accounts'] = credit2['accounts'].replace({
    'A14': 'No account',
    'A11': 'CC < 0',
    'A12': 'CC [0-200 euros[',
    'A13': 'CC > 200 euros'
})

credit2['history_credit'] = credit2['history_credit'].replace({
    'A30': 'Credits in arrears', 'A31': 'Credits in arrears',
    'A32': 'No credits or ongoing without delay', 'A33': 'No credits or ongoing without delay',
    'A34': 'Past credits without delay'
})

credit2['object_credit'] = credit2['object_credit'].replace({
    'A40': 'New car',
    'A41': 'Used car',
    'A42': 'Interior', 'A43': 'Interior', 'A45': 'Interior',
    'A46': 'Studies-business-Other', 'A48': 'Studies-business-Other', 'A410': 'Studies-business-Other',
    'A47': 'Vacations'
})

credit2['savings'] = credit2['savings'].replace({
    'A63': 'No savings or > 500 euros', 'A64': 'No savings or > 500 euros', 'A65': 'No savings or > 500 euros',
    'A61': '< 500 euros', 'A62': '< 500 euros'
})

credit2['employment_old'] = credit2['employment_old'].replace({
    'A71': 'Unemployed or < 1 year', 'A72': 'Unemployed or < 1 year',
    'A73': 'E [1-4[ years',
    'A74': 'E GE A years', 'A75': 'E GE A years'
})

credit2['family_status'] = credit2['family_status'].replace({
    'A91': 'Divorced/separated man',
    'A92': 'Divorced/separated/married woman',
    'A93': 'Single/married/widowed man', 'A94': 'Single/married/widowed man',
    'A95': 'Single woman'
})

credit2['guarantees'] = credit2['guarantees'].replace({'A103': 'With guarantor'})

credit2.loc[credit2['guarantees'] != 'With guarantor', 'guarantees'] = 'Without guarantor'

credit2['property'] = credit2['property'].replace({'A121': 'Real estate', 'A124': 'No assets'})

credit2.loc[~credit2['property'].isin(['Real estate', 'No assets']), 'property'] = 'Non-real estate'

credit2['other_credits'] = credit2['other_credits'].replace({'A143': 'No external credit'})

credit2.loc[credit2['other_credits'] != 'No external credit', 'other_credits'] = 'External credits'

credit2['home_status'] = credit2['home_status'].replace({'A152': 'Owner'})

credit2.loc[credit2['home_status'] != 'Owner', 'home_status'] = 'Non-owner' 


# splitting train and test 
test_size = 0.3

shuffled_data = credit2.sample(frac=1, random_state=42)

split_point = int(len(shuffled_data) * (1 - test_size))

train = shuffled_data.iloc[:split_point].reset_index(drop=True)
test = shuffled_data.iloc[split_point:].reset_index(drop=True)


####################### Model Explanation ##################################################################################################


if selected == "Model Explanation":

    # Get the summary
    st.header("How the Logit model is explained")
    st.subheader("Logit Model Summary")

    formula = 'presence_unpaid ~ accounts + history_credit + duration_credit + age + savings + guarantees + other_credits'
    logit_model = smf.glm(formula=formula, data=train, family=sm.families.Binomial(link=sm.genmod.families.links.Logit())).fit()

    st.text(logit_model.summary())


    # Get the metrics
    null_deviance = -2 * logit_model.llnull
    residual_deviance = -2 * logit_model.llf

    null_deviance_message = "Null Deviance: {:.4f}".format(null_deviance)
    residual_deviance_message = "Residual Deviance: {:.4f}".format(residual_deviance)
    aic_message =  "AIC: {:.4f}".format(logit_model.aic)
    fisher_message = "Number of Fisher Scoring iterations: {}".format(logit_model.fit_history['iteration'])

    st.write(null_deviance_message)
    st.write(residual_deviance_message)
    st.write(aic_message)
    st.write(fisher_message)

    predicted_probabilities = logit_model.predict(test)
    predicted_classes = (predicted_probabilities > 0.5).astype(int)

    y_test = test['presence_unpaid']

    auc = roc_auc_score(y_test, predicted_probabilities)
    auc_message = "Area Under The Curve ROC: {:.4f}".format(auc)
    st.write(auc_message)



    # Get Score Distribution Function
    st.header("How bad and good files are distributed")

    def ecdf(data):
        n = len(data)
        x = np.sort(data)
        y = np.arange(1, n+1) / n
        return x, y

    x_0, y_0 = ecdf(predicted_probabilities[test['presence_unpaid'] == 0])
    x_1, y_1 = ecdf(predicted_probabilities[test['presence_unpaid'] == 1])

    common_x = np.unique(np.concatenate((x_0, x_1)))
    y_0_interp = np.interp(common_x, x_0, y_0)
    y_1_interp = np.interp(common_x, x_1, y_1)

    ks_stat = np.max(np.abs(y_0_interp - y_1_interp))
    ks_x = common_x[np.argmax(np.abs(y_0_interp - y_1_interp))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_0, y=y_0, mode='markers', name='Score=0',
                            marker=dict(symbol='8', color='blue')))
    
    fig.add_trace(go.Scatter(x=x_1, y=y_1, mode='markers', name='Score=1',
                            marker=dict(symbol='triangle-up', color='red')))
    
    fig.add_trace(go.Scatter(x=[ks_x, ks_x], y=[y_0_interp[np.argmax(np.abs(y_0_interp - y_1_interp))], y_1_interp[np.argmax(np.abs(y_0_interp - y_1_interp))]],
                            mode='lines', name='KS Statistic', line=dict(color='grey', dash='dot')))
    fig.update_layout(
        title='Score Distribution Function',
        xaxis_title='x - Score (logit)',
        yaxis_title='Fn(x)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig)
  
    
    
    # Get Score Density Plot
    fpr, tpr, thresholds = roc_curve(y_test, predicted_probabilities)
    ks = tpr - fpr
    max_ks_index = np.argmax(ks)
    optimal_threshold = thresholds[max_ks_index]
    
    st.header("What is the density for bad and good files")

    scores_good = predicted_probabilities[y_test == 0]
    scores_bad = predicted_probabilities[y_test == 1]

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=scores_good, nbinsx=50, name='Good files (presence_unpaid=0)', opacity=0.7, histnorm='probability density', marker=dict(color='blue')))

    fig.add_trace(go.Histogram(x=scores_bad, nbinsx=50, name='Bad files (presence_unpaid=1)', opacity=0.7, histnorm='probability density', marker=dict(color='red')))

    fig.add_shape(type="line",
                x0=optimal_threshold, y0=0, x1=optimal_threshold, y1=1,
                line=dict(color="Green", width=3, dash="dash"))

    fig.add_annotation(x=optimal_threshold, y=0.5,
                    text=f'Optimum threshold: {optimal_threshold:.2f}',
                    showarrow=True, arrowhead=1)

    fig.update_layout(
        title="Score Density Functions with Optimal Threshold",
        xaxis_title="Score (predicted probability)",
        yaxis_title="Density",
        barmode='overlay'
    )

    st.plotly_chart(fig)


    # Get the boxplots
    st.header("How the good and bad files are well discriminated")

    test['predicted_score'] = predicted_probabilities

    fig = px.box(
        test,
        x='presence_unpaid',
        y='predicted_score',
        title='The score in each of the classes to be discriminated',
        labels={
            'presence_unpaid': 'Presence Unpaid',
            'predicted_score': 'Logit (Predicted Score)'
        },

        points="all",  
        color = 'presence_unpaid'
    )

    st.plotly_chart(fig)


    # Get the score grid
    st.header("How to put a weight on each modality's variable to measure its importance")
    st.subheader("Final Score Grid")

    model_logitxlevels = ['accounts', 'history_credit', 'duration_credit', 'age', 'savings','guarantees', 'other_credits']

    for var in model_logitxlevels:
        if var in credit2.columns:
            print(f"Level for {var}: {credit2[var].unique()}")
        else:
            print(f"{var} is not in the DataFrame")

    coefficients = logit_model.params


    multiplier = 10

    scores = {k: v * multiplier for k, v in coefficients.items() if k != 'Intercept'}

    print("Scores attributed to each modality :", scores)

    max_score = sum([abs(s) for s in scores.values()])  
    normalized_scores = {k: (v / max_score) * 100 for k, v in scores.items()}

    coefficients = logit_model.params

    data = {
        'VARIABLE': [],
        'MODALITY': [],
        'COEFF': []
    }

    for var in coefficients.index:
        if 'T.' in var:
            var_name, modality = var.split('T.')
        else:
            var_name = var
            modality = 'N/A'
    
        var_name = var_name.strip('[]').strip() 
        modality = modality.strip('[]').strip()  
        
        data['VARIABLE'].append(var_name)
        data['MODALITY'].append(modality)
        data['COEFF'].append(coefficients[var])

    param = pd.DataFrame(data)

    coefficients = logit_model.params

    min_coeffs = pd.DataFrame(coefficients.groupby(coefficients.index.str.split('[').str[0]).min(), columns=['Min_Coefficient'])
    max_coeffs = pd.DataFrame(coefficients.groupby(coefficients.index.str.split('[').str[0]).max(), columns=['Max_Coefficient'])

    total = min_coeffs.join(max_coeffs)
    total.reset_index(inplace=True)
    total.rename(columns={'index': 'VARIABLE'}, inplace=True)
    total['Diff'] = total['Max_Coefficient'] - total['Min_Coefficient']
    total_weight = sum(total['Diff'] )

    grid = pd.merge(param, min_coeffs, left_on='VARIABLE', right_index=True)
    grid['Modality_Weight'] = grid['COEFF'] - grid['Min_Coefficient']
    grid['WEIGHT'] = ((grid['Modality_Weight'] *100) / total_weight) 
    grid['WEIGHT'] = grid['WEIGHT'].round().astype(int)
   
    grid = grid[grid['VARIABLE'] != 'Intercept']
    grid.sort_values(by=['VARIABLE', 'MODALITY'], inplace=True)
    grid = grid[['VARIABLE', 'MODALITY', 'WEIGHT']]

    st.table(grid)

  

    # Get the ROC Curve
    st.header("How the logit model is efficient 1")
    
    from sklearn.metrics import roc_curve, auc as calculate_auc

    y_true = test['presence_unpaid']
    scores = predicted_probabilities
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc_value  = calculate_auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                            name=f'ROC curve (area = {roc_auc_value :.2f})',
                            line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                            name='Random', line=dict(color='grey', dash='dash')))

    fig.update_layout(
        title='ROC curve for the logistic model',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0.0, 1.0]),
        yaxis=dict(range=[0.0, 1.05]),
        showlegend=True
    )

    st.plotly_chart(fig)


    # Get Lift curve
    y_true = test['presence_unpaid']
    scores = predicted_probabilities

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)

    st.header("How the logit model is efficient 2")

    y_true_numeric = y_true.astype('int')
    prop_positive = y_true_numeric.mean()

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    rpp = fpr * (1 - prop_positive) + tpr * prop_positive  

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=rpp, y=tpr, mode='lines', 
                            name='Lift curve', line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=[0, prop_positive, 1], y=[0, 1, 1], mode='lines',
                            name='Perfect model', line=dict(color='grey', dash='dash')))

    fig.update_layout(
        title='Lift curve for the logistic model',
        xaxis_title='Rate of Positive Predictions (RPP)',
        yaxis_title='True Positive Rate (TPR)',
        showlegend=True
    )

    st.plotly_chart(fig)


    # Get the second Lift curve
    st.header("How the logit model is efficient 3")

    y_true_numeric = y_true.astype('int')
    prop_positive = y_true_numeric.mean()

    fpr, tpr, thresholds = roc_curve(y_true, scores)

    rpp = fpr * (1 - prop_positive) + tpr * prop_positive
    lift = tpr / rpp

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=rpp, y=lift, mode='lines', 
                            name='Lift curve',
                            line=dict(color='blue', width=2)))

    fig.add_trace(go.Scatter(x=[0, prop_positive, 1], y=[0, 1, 1], mode='lines',
                            name='Perfect model', line=dict(color='grey', dash='dash')))

    fig.update_layout(
        title='Lift curve for the logistic model',
        xaxis_title='Rate of Positive Predictions (RPP)',
        yaxis_title='Lift value',
        showlegend=True
    )

    st.plotly_chart(fig)




    # Get the Precision|Recall plot
    st.header("How to choose a threshold to maximise the performance of the model")

    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pr_thresholds, 
        y=precisions[:-1], 
        mode='lines', 
        name='Precision', 
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=pr_thresholds, 
        y=recalls[:-1], 
        mode='lines', 
        name='Recall', 
        line=dict(color='green')
    ))

    fig.update_layout(
        title='Precision and Recall vs Decision Threshold',
        xaxis_title='Decision Threshold',
        yaxis_title='Score',
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=20),
        template='plotly_white'
    )

    st.plotly_chart(fig)



    # Get the confusion matrix
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, scores)

    chosen_threshold = st.slider("Select Threshold", min_value=float(np.min(pr_thresholds)), 
                                max_value=float(np.max(pr_thresholds)), 
                                value=0.5, step=0.01)

    precision_value = 1  
    rounded_pr_thresholds = np.round(pr_thresholds, precision_value)
    rounded_chosen_threshold = np.round(chosen_threshold, precision_value)

    y_pred = (scores >= chosen_threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig = ff.create_annotated_heatmap(
        z=cm, 
        x=['Predicted Negative', 'Predicted Positive'], 
        y=['Actual Negative', 'Actual Positive'],
        annotation_text=[['TN: {}'.format(cm[0,0]), 'FP: {}'.format(cm[0,1])], 
                        ['FN: {}'.format(cm[1,0]), 'TP: {}'.format(cm[1,1])]],
        colorscale='Blues'
    )

    fig.update_layout(
        title=f'Confusion Matrix (Threshold = {chosen_threshold:.2f})',
        xaxis_title='Prediction',
        yaxis_title='Actual',
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig)


    # Get precision, recall and F1-score metrics
    if rounded_chosen_threshold in rounded_pr_thresholds:
        index = np.where(rounded_pr_thresholds == rounded_chosen_threshold)[0][0]
        precision = precisions[index]
        recall = recalls[index]
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        precision = "N/A"
        recall = "N/A"
        f1 = "N/A"
        st.write(f"The chosen threshold {rounded_chosen_threshold} does not exist in rounded pr_thresholds.")

    st.write(f'**Precision:** {precision * 100:.2f}%')
    st.write(f'**Recall:** {recall * 100:.2f}%')
    st.write(f'**F1 Score:** {f1 * 100:.2f}%')




