from train import main, set_random_seed
import torch


def grid_search(grid_search_params):
    for epoch in grid_search_params['num_epochs']:
        for lr in grid_search_params['learning_rate']:
            for num_lstm_layers in grid_search_params['num_lstm_layers']:
                for lstm_hidden_size in grid_search_params['lstm_hidden_size']:
                    print(f"Epoch: {epoch}, Learning Rate: {lr}, num_lstm_layers: {num_lstm_layers}, lstm_hidden_size: {lstm_hidden_size}")
                    participant_pids = [6, 7, 13, 14, 15, 16, 18]
                    metrics_dict = {
                        'participant_id': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'accuracy': []
                    }
                    params = {
                        'num_workers': grid_search_params['num_workers'],  # For data loading
                        'num_resnet_layers': grid_search_params['num_resnet_layers'],
                        'num_lstm_layers': num_lstm_layers,
                        'lstm_hidden_size': lstm_hidden_size,
                        'num_epochs': epoch,
                        'batch_size': grid_search_params['batch_size'],
                        'learning_rate': lr,
                        'data_dir': "/home/meixi/data",
                        'early_stop_patience': grid_search_params['early_stop_patience'],
                    }
                    for pid in participant_pids:
                        precision, recall, accuracy, f1 = main(params, pid, mode='grid_search')
                        metrics_dict['participant_id'].append(pid)
                        metrics_dict['precision'].append(round(precision, 4))
                        metrics_dict['recall'].append(round(recall, 4))
                        metrics_dict['f1_score'].append(round(f1, 4))
                        metrics_dict['accuracy'].append(round(accuracy, 4))
                    metrics_df = pd.DataFrame(metrics_dict)
                    metrics_df.set_index('participant_id', inplace=True)
                    avg_metrics = metrics_df.mean().round(4).to_frame().T
                    avg_metrics.index = ['Mean']
                    metrics_df = pd.concat([metrics_df, avg_metrics])
                    print(metrics_df)
                    print(f"----------------------------------")


if __name__ == '__main__':
    # Ensure reproducibility
    set_random_seed(42)
    g = torch.Generator()
    g.manual_seed(0)

    grid_search_params = {
        'num_workers': 4,  # For data loading
        'num_resnet_layers': 50,
        'batch_size': 512,
        'data_dir': "/home/meixi/data",
        'early_stop_patience': 5,
        'num_lstm_layers': [1, 2, 3],
        'lstm_hidden_size': [128, 256, 512],
        'num_epochs': [200, 500, 1000],   # Test to see if necessary
        'learning_rate': [0.0001, 0.00001, 0.000001],   # Test to see if necessary
    }
    grid_search(grid_search_params)