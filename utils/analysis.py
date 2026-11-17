"""
Analysis utilities for anomaly detection evaluation and statistics
"""
import os
import numpy as np
from .visualization import generate_overall_analysis_chart, generate_class_wise_analysis_charts


def analyze_classification_distribution(classification_scores, cls_names, anomaly_labels, save_dir):
    """
    Analyze classification distribution with detailed analysis by class
    Args:
        classification_scores: List of classification scores
        cls_names: List of class names
        anomaly_labels: List of anomaly labels (0=normal, 1=anomaly)
        save_dir: Save directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy arrays
    scores = np.array(classification_scores)
    labels = np.array(anomaly_labels)

    # Organize data by class
    class_data = {}
    for i, cls_name in enumerate(cls_names):
        if cls_name not in class_data:
            class_data[cls_name] = {'normal_scores': [], 'anomaly_scores': [], 'all_scores': []}
        class_data[cls_name]['all_scores'].append(scores[i])
        if labels[i] == 0:
            class_data[cls_name]['normal_scores'].append(scores[i])
        else:
            class_data[cls_name]['anomaly_scores'].append(scores[i])

    # Print overall statistics
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    print(f"\n📊 Classification Score Distribution Analysis:")
    print(f"Total samples: {len(scores)}")
    print(f"Normal samples: {len(normal_scores)} ({len(normal_scores)/len(scores)*100:.1f}%)")
    print(f"Anomaly samples: {len(anomaly_scores)} ({len(anomaly_scores)/len(scores)*100:.1f}%)")

    if len(normal_scores) > 0:
        print(f"\nOverall normal sample score statistics:")
        print(f"  Mean: {normal_scores.mean():.3f}, Std: {normal_scores.std():.3f}")
        print(f"  Min: {normal_scores.min():.3f}, Max: {normal_scores.max():.3f}")
        print(f"  Median: {np.median(normal_scores):.3f}")

    if len(anomaly_scores) > 0:
        print(f"\nOverall anomaly sample score statistics:")
        print(f"  Mean: {anomaly_scores.mean():.3f}, Std: {anomaly_scores.std():.3f}")
        print(f"  Min: {anomaly_scores.min():.3f}, Max: {anomaly_scores.max():.3f}")
        print(f"  Median: {np.median(anomaly_scores):.3f}")

    # Detailed analysis by class
    print(f"\n📈 Detailed Analysis by Class:")
    threshold = 0.0
    class_stats = []

    for cls_name in sorted(class_data.keys()):
        data = class_data[cls_name]
        normal_scores_cls = np.array(data['normal_scores'])
        anomaly_scores_cls = np.array(data['anomaly_scores'])
        all_scores_cls = np.array(data['all_scores'])

        total_samples = len(all_scores_cls)
        normal_count = len(normal_scores_cls)
        anomaly_count = len(anomaly_scores_cls)

        # Compute classification bias
        predicted_anomaly = all_scores_cls > threshold
        anomaly_ratio = np.mean(predicted_anomaly)
        bias = "More anomaly" if anomaly_ratio > 0.5 else "More normal"

        print(f"\n🏷️  {cls_name}:")
        print(f"   Total samples: {total_samples}")
        print(f"   Normal: {normal_count} ({normal_count/total_samples*100:.1f}%), Anomaly: {anomaly_count} ({anomaly_count/total_samples*100:.1f}%)")

        if len(normal_scores_cls) > 0:
            print(f"   Normal sample scores: mean={normal_scores_cls.mean():.3f}, std={normal_scores_cls.std():.3f}")
        if len(anomaly_scores_cls) > 0:
            print(f"   Anomaly sample scores: mean={anomaly_scores_cls.mean():.3f}, std={anomaly_scores_cls.std():.3f}")

        print(f"   Classification bias: {anomaly_ratio:.3f} ({bias})")

        class_stats.append({
            'name': cls_name,
            'total': total_samples,
            'normal_count': normal_count,
            'anomaly_count': anomaly_count,
            'normal_scores': normal_scores_cls,
            'anomaly_scores': anomaly_scores_cls,
            'all_scores': all_scores_cls,
            'anomaly_ratio': anomaly_ratio,
            'bias': bias
        })

    # Generate overall analysis chart
    generate_overall_analysis_chart(normal_scores, anomaly_scores, class_stats, save_dir)

    # Generate detailed analysis charts for each class
    generate_class_wise_analysis_charts(class_stats, save_dir)

    # Print classification bias summary
    print(f"\n🎯 Classification Bias Summary (threshold={threshold}):")
    print(f"{'Class':<15} {'Samples':<8} {'Anomaly Ratio':<12} {'Bias'}")
    print("-" * 50)

    # Sort by anomaly ratio
    sorted_stats = sorted(class_stats, key=lambda x: x['anomaly_ratio'], reverse=True)
    for stat in sorted_stats:
        print(f"{stat['name']:<15} {stat['total']:<8} {stat['anomaly_ratio']:<12.3f} {stat['bias']}")

    overall_anomaly_ratio = np.mean(scores > threshold)
    overall_bias = "More anomaly" if overall_anomaly_ratio > 0.5 else "More normal"
    print(f"{'Overall':<15} {len(scores):<8} {overall_anomaly_ratio:<12.3f} {overall_bias}")

    print(f"\n✅ Analysis charts saved to: {save_dir}/")
