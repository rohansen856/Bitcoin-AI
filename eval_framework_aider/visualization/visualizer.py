"""
Visualization and reporting utilities for benchmark results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List

from models.data_models import BenchmarkResult
from utils.logger import get_logger

logger = get_logger(__name__)


class BenchmarkVisualizer:
    """Generate visualizations for benchmark results"""
    
    def __init__(self, results: List[BenchmarkResult], output_dir: Path):
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Convert results to DataFrame
        self.df = self._results_to_dataframe()
        
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easier analysis"""
        return pd.DataFrame([
            {
                'model': r.model,
                'exercise': r.exercise,
                'language': r.language,
                'success': r.success,
                'execution_time': r.execution_time,
                'token_count': r.token_count,
                'timestamp': r.timestamp
            }
            for r in self.results
        ])
    
    def create_all_visualizations(self) -> List[Path]:
        """Create all visualization charts"""
        if self.df.empty:
            logger.warning("No data to visualize")
            return []
        
        logger.info("Generating visualizations...")
        
        created_files = []
        
        # Generate different visualizations
        created_files.extend([
            self.create_success_rate_by_model(),
            self.create_success_rate_by_language(),
            self.create_execution_time_comparison(),
            self.create_token_usage_analysis(),
            self.create_heatmap_model_exercise(),
            self.create_performance_radar_chart(),
            self.create_combined_dashboard()
        ])
        
        # Filter out None values
        created_files = [f for f in created_files if f is not None]
        
        logger.info(f"Generated {len(created_files)} visualizations in: {self.output_dir}")
        return created_files
    
    def create_success_rate_by_model(self) -> Path:
        """Bar chart of success rates by model"""
        plt.figure(figsize=(12, 6))
        
        # Calculate success rates
        success_rates = self.df.groupby('model')['success'].agg(['sum', 'count'])
        success_rates['rate'] = (success_rates['sum'] / success_rates['count']) * 100
        
        # Create bar chart
        bars = plt.bar(success_rates.index, success_rates['rate'])
        
        # Color bars based on performance
        colors = ['#2ecc71' if rate >= 80 else '#f39c12' if rate >= 60 else '#e74c3c' 
                  for rate in success_rates['rate']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Success Rate by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.ylim(0, 105)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = self.output_dir / 'success_rate_by_model.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_success_rate_by_language(self) -> Path:
        """Bar chart of success rates by programming language"""
        plt.figure(figsize=(10, 6))
        
        # Calculate success rates by language
        lang_success = self.df.groupby('language')['success'].agg(['sum', 'count'])
        lang_success['rate'] = (lang_success['sum'] / lang_success['count']) * 100
        
        # Create horizontal bar chart
        bars = plt.barh(lang_success.index, lang_success['rate'])
        
        # Apply gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.title('Success Rate by Programming Language', fontsize=16, fontweight='bold')
        plt.xlabel('Success Rate (%)', fontsize=12)
        plt.ylabel('Language', fontsize=12)
        plt.xlim(0, 105)
        
        # Add value labels
        for i, (idx, row) in enumerate(lang_success.iterrows()):
            plt.text(row['rate'] + 1, i, f"{row['rate']:.1f}%", 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'success_rate_by_language.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_execution_time_comparison(self) -> Path:
        """Box plot of execution times by model"""
        plt.figure(figsize=(12, 6))
        
        # Create box plot
        models = self.df['model'].unique()
        data_to_plot = [self.df[self.df['model'] == model]['execution_time'].values 
                        for model in models]
        
        bp = plt.boxplot(data_to_plot, labels=models, patch_artist=True)
        
        # Customize box plot colors
        colors = sns.color_palette("husl", len(models))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Execution Time Distribution by Model', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Add median values
        medians = [np.median(data) for data in data_to_plot]
        for i, median in enumerate(medians):
            plt.text(i + 1, median + 0.5, f'{median:.2f}s', 
                    ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / 'execution_time_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_token_usage_analysis(self) -> Path:
        """Scatter plot of tokens vs execution time"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot for each model
        models = self.df['model'].unique()
        colors = sns.color_palette("husl", len(models))
        
        for model, color in zip(models, colors):
            model_data = self.df[self.df['model'] == model]
            success_data = model_data[model_data['success']]
            fail_data = model_data[~model_data['success']]
            
            # Plot successful runs
            if not success_data.empty:
                plt.scatter(success_data['token_count'], success_data['execution_time'], 
                           label=f'{model} (success)', color=color, s=100, alpha=0.7, marker='o')
            
            # Plot failed runs
            if not fail_data.empty:
                plt.scatter(fail_data['token_count'], fail_data['execution_time'], 
                           label=f'{model} (failed)', color=color, s=100, alpha=0.7, marker='x')
        
        plt.title('Token Usage vs Execution Time', fontsize=16, fontweight='bold')
        plt.xlabel('Token Count', fontsize=12)
        plt.ylabel('Execution Time (seconds)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add trend line if we have enough data
        if len(self.df) > 1:
            x = self.df['token_count']
            y = self.df['execution_time']
            if x.var() > 0:  # Check if there's variation in x
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.5, label='Trend')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = self.output_dir / 'token_usage_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_heatmap_model_exercise(self) -> Path:
        """Heatmap showing success/failure for each model-exercise combination"""
        plt.figure(figsize=(14, 8))
        
        # Check if we have enough data for a meaningful heatmap
        if len(self.df['model'].unique()) < 2 or len(self.df['exercise'].unique()) < 2:
            logger.warning("Not enough data for heatmap")
            plt.close()
            return None
        
        # Pivot the data for heatmap
        pivot_data = self.df.pivot_table(
            values='success', 
            index='exercise', 
            columns='model', 
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt='.0f', 
                   cmap='RdYlGn', 
                   center=50,
                   cbar_kws={'label': 'Success Rate (%)'},
                   linewidths=0.5,
                   square=True)
        
        plt.title('Model Performance Heatmap by Exercise', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Exercise', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filepath = self.output_dir / 'heatmap_model_exercise.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_performance_radar_chart(self) -> Path:
        """Radar chart comparing models across multiple metrics"""
        models = self.df['model'].unique()
        if len(models) < 2:
            logger.warning("Not enough models for radar chart")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate metrics for each model
        metrics = ['Success Rate', 'Avg Speed', 'Token Efficiency', 'Consistency']
        
        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, model in enumerate(models):
            model_data = self.df[self.df['model'] == model]
            
            # Calculate metrics (normalized to 0-100)
            success_rate = (model_data['success'].sum() / len(model_data)) * 100
            avg_speed = 100 - min((model_data['execution_time'].mean() / 10) * 100, 100)
            token_efficiency = 100 - min((model_data['token_count'].mean() / 1000) * 100, 100)
            consistency = 100 - min((model_data['execution_time'].std() / 
                                   model_data['execution_time'].mean()) * 100, 100)
            
            values = [success_rate, avg_speed, token_efficiency, consistency]
            values += values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        # Customize chart
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 100)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])
        ax.grid(True)
        
        plt.title('Model Performance Radar Chart', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        
        filepath = self.output_dir / 'performance_radar_chart.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def create_combined_dashboard(self) -> Path:
        """Create a comprehensive dashboard with multiple visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Summary (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_summary_stats(ax1)
        
        # 2. Success rate by model (top-middle and top-right)
        ax2 = fig.add_subplot(gs[0, 1:])
        self._create_model_comparison_bars(ax2)
        
        # 3. Language performance (middle-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._create_language_pie_chart(ax3)
        
        # 4. Time distribution (middle-center)
        ax4 = fig.add_subplot(gs[1, 1])
        self._create_time_distribution(ax4)
        
        # 5. Token distribution (middle-right)
        ax5 = fig.add_subplot(gs[1, 2])
        self._create_token_distribution(ax5)
        
        # 6. Performance matrix (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        self._create_performance_matrix(ax6)
        
        plt.suptitle('LLM Code Generation Benchmark Dashboard', fontsize=20, fontweight='bold')
        
        filepath = self.output_dir / 'benchmark_dashboard.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def _create_summary_stats(self, ax):
        """Create summary statistics panel"""
        ax.axis('off')
        
        # Calculate overall stats
        total_tests = len(self.df)
        total_passed = self.df['success'].sum()
        overall_success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
        avg_time = self.df['execution_time'].mean()
        avg_tokens = self.df['token_count'].mean()
        
        # Create text summary
        summary_text = f"""
        OVERALL STATISTICS
        
        Total Tests: {total_tests}
        Passed: {total_passed}
        Failed: {total_tests - total_passed}
        
        Success Rate: {overall_success_rate:.1f}%
        Avg Time: {avg_time:.2f}s
        Avg Tokens: {avg_tokens:.0f}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=14, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
        
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')
    
    def _create_model_comparison_bars(self, ax):
        """Create grouped bar chart for model comparison"""
        model_stats = self.df.groupby('model').agg({
            'success': ['sum', 'count'],
            'execution_time': 'mean',
            'token_count': 'mean'
        })
        
        model_stats['success_rate'] = (model_stats['success']['sum'] / 
                                     model_stats['success']['count']) * 100
        
        x = np.arange(len(model_stats.index))
        width = 0.35
        
        # Success rate bars
        bars1 = ax.bar(x - width/2, model_stats['success_rate'], width, 
                      label='Success Rate (%)')
        
        # Normalized execution time (inverted for better visualization)
        max_time = model_stats['execution_time']['mean'].max()
        if max_time > 0:
            normalized_time = (1 - model_stats['execution_time']['mean'] / max_time) * 100
            bars2 = ax.bar(x + width/2, normalized_time, width, label='Speed Score')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_stats.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_language_pie_chart(self, ax):
        """Create pie chart for language distribution"""
        lang_counts = self.df.groupby('language')['success'].agg(['sum', 'count'])
        
        if len(lang_counts) > 0:
            colors = plt.cm.Set3(np.linspace(0, 1, len(lang_counts)))
            wedges, texts, autotexts = ax.pie(lang_counts['count'], 
                                              labels=lang_counts.index,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              startangle=90)
        
        ax.set_title('Test Distribution by Language', fontsize=14, fontweight='bold')
    
    def _create_time_distribution(self, ax):
        """Create histogram of execution times"""
        if len(self.df) > 0:
            ax.hist(self.df['execution_time'], bins=20, color='skyblue', 
                   edgecolor='black', alpha=0.7)
            mean_time = self.df['execution_time'].mean()
            ax.axvline(mean_time, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {mean_time:.2f}s')
            ax.legend()
        
        ax.set_xlabel('Execution Time (s)')
        ax.set_ylabel('Frequency')
        ax.set_title('Execution Time Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _create_token_distribution(self, ax):
        """Create histogram of token counts"""
        if len(self.df) > 0:
            ax.hist(self.df['token_count'], bins=20, color='lightgreen', 
                   edgecolor='black', alpha=0.7)
            mean_tokens = self.df['token_count'].mean()
            ax.axvline(mean_tokens, color='red', linestyle='dashed', linewidth=2, 
                      label=f'Mean: {mean_tokens:.0f}')
            ax.legend()
        
        ax.set_xlabel('Token Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Token Usage Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _create_performance_matrix(self, ax):
        """Create a performance matrix visualization"""
        models = sorted(self.df['model'].unique())
        exercises = sorted(self.df['exercise'].unique())
        
        if len(models) == 0 or len(exercises) == 0:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Performance Matrix', fontsize=14, fontweight='bold')
            return
        
        matrix = np.zeros((len(models), len(exercises)))
        
        for i, model in enumerate(models):
            for j, exercise in enumerate(exercises):
                result = self.df[(self.df['model'] == model) & 
                               (self.df['exercise'] == exercise)]
                if not result.empty:
                    matrix[i, j] = 1 if result.iloc[0]['success'] else -1
        
        # Custom colormap: red for fail, green for pass, white for no data
        colors = ['red', 'white', 'green']
        n_bins = 3
        cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(exercises)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(exercises, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        cbar.ax.set_yticklabels(['Failed', 'No Data', 'Passed'])
        
        ax.set_title('Model-Exercise Performance Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Exercise')
        ax.set_ylabel('Model')
        
        # Add grid
        ax.set_xticks(np.arange(len(exercises)+1)-.5, minor=True)
        ax.set_yticks(np.arange(len(models)+1)-.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)


class QuickVisualizer:
    """Quick visualization utilities for simple charts"""
    
    @staticmethod
    def create_quick_comparison(results: List[BenchmarkResult], output_path: Path) -> Path:
        """Create a quick comparison chart"""
        if not results:
            return None
        
        df = pd.DataFrame([
            {
                'model': r.model,
                'success': r.success,
                'execution_time': r.execution_time
            }
            for r in results
        ])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate by model
        success_rates = df.groupby('model')['success'].mean() * 100
        ax1.bar(success_rates.index, success_rates.values)
        ax1.set_title('Success Rate by Model')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_xlabel('Model')
        
        # Execution time by model
        df.boxplot(column='execution_time', by='model', ax=ax2)
        ax2.set_title('Execution Time by Model')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_xlabel('Model')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    @staticmethod
    def create_progress_chart(results: List[BenchmarkResult], output_path: Path) -> Path:
        """Create a progress/timeline chart"""
        if not results:
            return None
        
        df = pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'model': r.model,
                'success': r.success
            }
            for r in results
        ])
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Calculate cumulative success rate
        df['cumulative_success'] = df['success'].cumsum()
        df['cumulative_total'] = range(1, len(df) + 1)
        df['cumulative_rate'] = (df['cumulative_success'] / df['cumulative_total']) * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['cumulative_rate'], marker='o', linewidth=2)
        plt.title('Cumulative Success Rate Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Success Rate (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path