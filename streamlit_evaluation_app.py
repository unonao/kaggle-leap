import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from pathlib import Path
import pickle

st.set_page_config(
    page_title="Atmospheric Physics Simulation Evaluation",
    page_icon="🌍",
    layout="wide"
)

def generate_dummy_data():
    """Generate dummy data mimicking the atmospheric physics simulation format"""
    
    single_targets = [
        "cam_out_NETSW",
        "cam_out_FLWDS", 
        "cam_out_PRECSC",
        "cam_out_PRECC",
        "cam_out_SOLS",
        "cam_out_SOLL",
        "cam_out_SOLSD",
        "cam_out_SOLLD",
    ]
    
    seq_targets = [
        "ptend_t",
        "ptend_q0001", 
        "ptend_q0002",
        "ptend_q0003",
        "ptend_u",
        "ptend_v",
    ]
    
    target_columns = []
    for col in seq_targets:
        for i in range(60):
            target_columns.append(f"{col}_{i}")
    target_columns.extend(single_targets)
    
    n_samples = 1000
    n_targets = len(target_columns)
    
    np.random.seed(42)
    
    ground_truth = np.zeros((n_samples, n_targets))
    predictions = np.zeros((n_samples, n_targets))
    
    col_idx = 0
    
    for seq_target in seq_targets:
        for level in range(60):
            if seq_target == "ptend_t":  # Temperature tendency
                base_pattern = np.sin(level * np.pi / 60) * 0.1
                noise_scale = 0.02
            elif seq_target.startswith("ptend_q"):  # Humidity tendencies
                base_pattern = np.exp(-level / 20) * 0.05
                noise_scale = 0.01
            elif seq_target in ["ptend_u", "ptend_v"]:  # Wind tendencies
                base_pattern = np.cos(level * np.pi / 30) * 0.08
                noise_scale = 0.015
            else:
                base_pattern = 0.01
                noise_scale = 0.005
                
            ground_truth[:, col_idx] = base_pattern + np.random.normal(0, noise_scale, n_samples)
            
            correlation = 0.7 + 0.2 * np.random.random()  # Random correlation between 0.7-0.9
            predictions[:, col_idx] = (correlation * ground_truth[:, col_idx] + 
                                     (1 - correlation) * np.random.normal(0, noise_scale, n_samples))
            col_idx += 1
    
    for single_target in single_targets:
        if "PREC" in single_target:  # Precipitation
            ground_truth[:, col_idx] = np.abs(np.random.gamma(2, 0.1, n_samples))
            correlation = 0.6 + 0.3 * np.random.random()
        elif "NET" in single_target or "FL" in single_target:  # Radiation fluxes
            ground_truth[:, col_idx] = 200 + 100 * np.random.normal(0, 1, n_samples)
            correlation = 0.8 + 0.15 * np.random.random()
        else:  # Solar radiation
            ground_truth[:, col_idx] = 300 + 150 * np.random.normal(0, 1, n_samples)
            correlation = 0.75 + 0.2 * np.random.random()
            
        predictions[:, col_idx] = (correlation * ground_truth[:, col_idx] + 
                                 (1 - correlation) * np.random.normal(ground_truth[:, col_idx].mean(), 
                                                                     ground_truth[:, col_idx].std(), n_samples))
        col_idx += 1
    
    ground_truth_df = pd.DataFrame(ground_truth, columns=target_columns)
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    
    ground_truth_df.insert(0, 'sample_id', range(n_samples))
    predictions_df.insert(0, 'sample_id', range(n_samples))
    
    return ground_truth_df, predictions_df, target_columns, seq_targets, single_targets

def calculate_r2_scores(ground_truth_df, predictions_df, target_columns):
    """Calculate R2 scores for all target columns"""
    r2_scores = {}
    
    for col in target_columns:
        y_true = ground_truth_df[col].values
        y_pred = predictions_df[col].values
        r2_scores[col] = r2_score(y_true, y_pred)
    
    return r2_scores

def plot_sequential_targets(r2_scores, seq_targets):
    """Plot R2 scores for sequential targets"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(seq_targets):
        if idx < len(axes):
            ax = axes[idx]
            
            r2_values = [r2_scores[f"{col}_{i}"] for i in range(60)]
            
            ax.plot(range(60), r2_values, 'o-', linewidth=2, markersize=4, label='R2 scores')
            ax.set_title(f'{col} - R2 Scores by Level', fontsize=12, fontweight='bold')
            ax.set_xlabel('Vertical Level')
            ax.set_ylabel('R2 Score')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.1, 1.1)
            
            mean_r2 = np.mean(r2_values)
            ax.axhline(y=mean_r2, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean R2: {mean_r2:.3f}')
            ax.legend()
    
    for idx in range(len(seq_targets), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_single_targets(r2_scores, single_targets):
    """Plot R2 scores for single targets"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    r2_values = [r2_scores[col] for col in single_targets]
    
    bars = ax.bar(single_targets, r2_values, color='skyblue', alpha=0.8, edgecolor='navy')
    ax.set_title('R2 Scores for Single Targets', fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Variables')
    ax.set_ylabel('R2 Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_interactive_scatter_plot(ground_truth_df, predictions_df, target_columns):
    """Create interactive scatter plot for prediction vs ground truth"""
    
    selected_targets = [
        "ptend_t_30",  # Mid-level temperature tendency
        "ptend_q0001_10",  # Low-level humidity tendency
        "cam_out_PRECC",  # Convective precipitation
        "cam_out_NETSW"   # Net shortwave radiation
    ]
    
    fig = go.Figure()
    
    for target in selected_targets:
        if target in target_columns:
            y_true = ground_truth_df[target].values
            y_pred = predictions_df[target].values
            
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name=target,
                opacity=0.6,
                hovertemplate=f'<b>{target}</b><br>' +
                             'Ground Truth: %{x:.4f}<br>' +
                             'Prediction: %{y:.4f}<br>' +
                             '<extra></extra>'
            ))
    
    all_values = []
    for target in selected_targets:
        if target in target_columns:
            all_values.extend(ground_truth_df[target].values)
            all_values.extend(predictions_df[target].values)
    
    min_val, max_val = min(all_values), max(all_values)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash'),
        hovertemplate='Perfect Prediction Line<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prediction vs Ground Truth Scatter Plot',
        xaxis_title='Ground Truth',
        yaxis_title='Predictions',
        hovermode='closest',
        height=600
    )
    
    return fig

def main():
    st.title("🌍 Atmospheric Physics Simulation Evaluation Dashboard")
    st.markdown("---")
    
    st.sidebar.header("Dashboard Controls")
    
    if st.sidebar.button("Generate New Dummy Data"):
        st.session_state.data_generated = True
    
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = True
    
    if st.session_state.data_generated:
        with st.spinner("Generating dummy atmospheric physics data..."):
            ground_truth_df, predictions_df, target_columns, seq_targets, single_targets = generate_dummy_data()
            r2_scores = calculate_r2_scores(ground_truth_df, predictions_df, target_columns)
        
        st.success("✅ Dummy data generated successfully!")
        
        st.header("📊 Overall Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            overall_r2 = np.mean(list(r2_scores.values()))
            st.metric("Overall R2 Score", f"{overall_r2:.4f}")
        
        with col2:
            seq_r2 = np.mean([r2_scores[f"{col}_{i}"] for col in seq_targets for i in range(60)])
            st.metric("Sequential Targets R2", f"{seq_r2:.4f}")
        
        with col3:
            single_r2 = np.mean([r2_scores[col] for col in single_targets])
            st.metric("Single Targets R2", f"{single_r2:.4f}")
        
        with col4:
            n_samples = len(ground_truth_df)
            st.metric("Number of Samples", f"{n_samples:,}")
        
        st.markdown("---")
        
        st.header("🌡️ Sequential Targets Analysis")
        st.markdown("Sequential targets represent vertical profiles (60 levels) for atmospheric variables like temperature and humidity tendencies.")
        
        with st.spinner("Creating sequential targets visualization..."):
            seq_fig = plot_sequential_targets(r2_scores, seq_targets)
            st.pyplot(seq_fig)
        
        st.subheader("Detailed Sequential Target Performance")
        selected_seq_target = st.selectbox("Select Sequential Target for Detailed View:", seq_targets)
        
        if selected_seq_target:
            seq_r2_values = [r2_scores[f"{selected_seq_target}_{i}"] for i in range(60)]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(range(60), seq_r2_values, 'o-', linewidth=2, markersize=4)
            ax1.set_title(f'{selected_seq_target} - R2 by Vertical Level')
            ax1.set_xlabel('Vertical Level')
            ax1.set_ylabel('R2 Score')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.1, 1.1)
            
            ax2.hist(seq_r2_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_title(f'{selected_seq_target} - R2 Score Distribution')
            ax2.set_xlabel('R2 Score')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean R2", f"{np.mean(seq_r2_values):.4f}")
            with col2:
                st.metric("Std R2", f"{np.std(seq_r2_values):.4f}")
            with col3:
                st.metric("Min R2", f"{np.min(seq_r2_values):.4f}")
            with col4:
                st.metric("Max R2", f"{np.max(seq_r2_values):.4f}")
        
        st.markdown("---")
        
        st.header("☀️ Single Targets Analysis")
        st.markdown("Single targets represent surface fluxes, radiation, and precipitation variables.")
        
        with st.spinner("Creating single targets visualization..."):
            single_fig = plot_single_targets(r2_scores, single_targets)
            st.pyplot(single_fig)
        
        st.subheader("Single Target Performance Details")
        single_target_data = []
        for target in single_targets:
            single_target_data.append({
                'Target': target,
                'R2 Score': r2_scores[target],
                'Mean Ground Truth': ground_truth_df[target].mean(),
                'Mean Prediction': predictions_df[target].mean(),
                'Ground Truth Std': ground_truth_df[target].std(),
                'Prediction Std': predictions_df[target].std()
            })
        
        single_df = pd.DataFrame(single_target_data)
        st.dataframe(single_df.round(4), use_container_width=True)
        
        st.markdown("---")
        
        st.header("🎯 Interactive Prediction Analysis")
        st.markdown("Explore the relationship between predictions and ground truth for selected targets.")
        
        with st.spinner("Creating interactive scatter plot..."):
            scatter_fig = create_interactive_scatter_plot(ground_truth_df, predictions_df, target_columns)
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        st.markdown("---")
        
        st.header("🔍 Data Exploration")
        
        tab1, tab2, tab3 = st.tabs(["Ground Truth Data", "Predictions Data", "R2 Scores"])
        
        with tab1:
            st.subheader("Ground Truth Sample")
            st.dataframe(ground_truth_df.head(10), use_container_width=True)
            
            st.subheader("Ground Truth Statistics")
            st.dataframe(ground_truth_df.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("Predictions Sample")
            st.dataframe(predictions_df.head(10), use_container_width=True)
            
            st.subheader("Predictions Statistics")
            st.dataframe(predictions_df.describe(), use_container_width=True)
        
        with tab3:
            st.subheader("All R2 Scores")
            r2_df = pd.DataFrame(list(r2_scores.items()), columns=['Target', 'R2 Score'])
            r2_df = r2_df.sort_values('R2 Score', ascending=False)
            st.dataframe(r2_df, use_container_width=True)
            
            csv = r2_df.to_csv(index=False)
            st.download_button(
                label="Download R2 Scores as CSV",
                data=csv,
                file_name="r2_scores.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        st.markdown("### 📝 About This Dashboard")
        st.markdown("""
        This evaluation dashboard simulates the analysis of atmospheric physics simulation results:
        
        - **Sequential Targets**: Represent vertical atmospheric profiles (60 levels) for variables like temperature and humidity tendencies
        - **Single Targets**: Represent surface-level variables like radiation fluxes and precipitation
        - **R2 Score**: Coefficient of determination used as the primary evaluation metric
        - **Dummy Data**: Realistic synthetic data mimicking atmospheric physics patterns
        
        The dashboard provides comprehensive evaluation capabilities including:
        - Overall performance metrics
        - Detailed analysis by target type
        - Interactive visualizations
        - Data exploration tools
        """)

if __name__ == "__main__":
    main()
