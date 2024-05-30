# NeuralNetwork
This is a PHP package for building neural networks in PHP, if theirs enough interest I will adapt it into a PHP C module providing very quick training on the CPU and GPU. 

### Example Usage

```
 $total_hidden_neurons = 4;
    $total_hidden_layers = 2;
    $learning_rate = 0.1;
    $epochs = 500000;
    $total_input_neurons = 2;
    $total_output_neurons = 1;

    // You can use either "relu" or "sigmoid" for the activation function
    $network = new NeuralNetwork($total_input_neurons, $total_output_neurons, $learning_rate, $epochs, $total_hidden_neurons, $total_hidden_layers, 'relu');

    // Train the network on one page at a time, useful for large datasets
    // where you can simply load from the database.
    $network->trainNetworkBatch(4, function ($page, $total_per_page, &$input, &$output) {
        switch ($page) {
            case 0:
                $input = [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                ];
                $output = [
                    [0], [1], [1]
                ];
                break;
            case 1:
                $input = [
                    [1, 1]
                ];
                $output = [
                    [0]
                ];
                break;
          
            default:
                echo 'out of bounds';
                exit;
                break;
        }
    }, 3);


    $network->forwardPass([1, 0], $output);
    print_r($output);
```

