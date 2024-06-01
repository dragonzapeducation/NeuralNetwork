<?php

namespace Dragonzap\NeuralNetwork;
use Dragonzap\Exceptions\BadNetworkParameterException;
use Dragonzap\Exceptions\CriticalTrainingErrorException;

class NeuralNetwork
{
    protected $total_input_neurons = 2;
    protected $total_hidden_layer_neurons = 4;
    protected $total_hidden_layers = 1;
    protected $total_output_neurons = 1;
    protected $learningRate = 0.1;
    protected $epochs = 1000;

    protected $activation_function = 'sigmoid';
    protected $activation_function_derivative = 'sigmoidDerivative';
    public $layers = [];

    public function __construct($total_input_neurons, $total_output_neurons, $learning_rate = 0.1, $epochs = 100000, $total_hidden_layer_neurons = 128, $total_hidden_layers = 1, $activation_function = 'sigmoid')
    {
        $this->total_input_neurons = $total_input_neurons;
        $this->total_output_neurons = $total_output_neurons;
        $this->learningRate = $learning_rate;
        $this->epochs = $epochs;
        $this->total_hidden_layer_neurons = $total_hidden_layer_neurons;
        $this->total_hidden_layers = $total_hidden_layers;
        $this->activation_function = $activation_function;
        $this->activation_function_derivative = $activation_function . 'Derivative';
        if ($this->validActivationFunction($activation_function) === false) {
            throw new BadNetworkParameterException('Invalid activation function');
        }
        $this->initializeLayers();
    }


    public function getTotalInputNeurons()
    {
        return $this->total_input_neurons;
    }

    public function getTotalOutputNeurons()
    {
        return $this->total_input_neurons;
    }

    public function getLayers()
    {
        return $this->layers;
    }


    private function initializeLayers()
    {
        for ($i = 0; $i < $this->total_hidden_layers; $i++) {
            $total_last_neurons = $i == 0 ? $this->total_input_neurons : count($this->layers[$i - 1]->neurons);
            $this->layers[] = new Layer($this->total_hidden_layer_neurons, $total_last_neurons);
        }

        $last_layer_neurons = count($this->layers[count($this->layers) - 1]->neurons);
        $this->layers[] = new Layer($this->total_output_neurons, $last_layer_neurons);
    }

    private function validActivationFunction($function)
    {
        return in_array($function, ['sigmoid', 'relu']);
    }

    private function relu($x)
    {
        return max(0, $x);
    }

    private function reluDerivative($x)
    {
        return $x > 0 ? 1 : 0;
    }

    private function sigmoid($x)
    {
        return 1 / (1 + exp(-$x));
    }

    private function sigmoidDerivative($x)
    {
        return $x * (1 - $x);
    }

    public function forwardPass($input, &$output)
    {
        $input_to_layer = $input;

        foreach ($this->layers as $layer) {
            $layer_output = [];
            foreach ($layer->neurons as $neuron) {
                $neuron_output = 0;
                foreach ($neuron->weights as $weight_index => $weight) {
                    $neuron_output += $input_to_layer[$weight_index] * $weight;
                }
                $neuron->output = call_user_func([$this, $this->activation_function], $neuron_output);
                $layer_output[] = $neuron->output;
            }
            $input_to_layer = $layer_output;
            $layer->output = $layer_output;
        }

        $output = $input_to_layer;
    }

    private function trainNetworkEpoch(array $input_data, array $expected_output)
    {
        foreach ($input_data as $index => $input) {
            $output = [];
            $this->forwardPass($input, $output);

            $output_layer = end($this->layers);
            foreach ($output_layer->neurons as $output_index => $neuron) {
                $error = $expected_output[$index][$output_index] - $output[$output_index];
                $neuron->delta = $error * call_user_func([$this, $this->activation_function_derivative], $neuron->output);
            }

            for ($i = count($this->layers) - 2; $i >= 0; $i--) {
                $current_layer = $this->layers[$i];
                $next_layer = $this->layers[$i + 1];

                foreach ($current_layer->neurons as $neuron_index => $neuron) {
                    $error = 0;
                    foreach ($next_layer->neurons as $next_neuron) {
                        $error += $next_neuron->delta * $next_neuron->weights[$neuron_index];
                    }
                    $neuron->delta = $error * call_user_func([$this, $this->activation_function_derivative], $neuron->output);
                }
            }

            $input_to_layer = $input;
            foreach ($this->layers as $layer) {
                foreach ($layer->neurons as $neuron) {
                    foreach ($neuron->weights as $weight_index => &$weight) {
                        $weight += $this->learningRate * $neuron->delta * $input_to_layer[$weight_index];
                    }
                }
                $input_to_layer = $layer->output;
            }
        }
    }

    public function trainNetworkBatch(int $total_records, $read_records_callback, $total_per_page = 100)
    {
        if ($total_records == 0) {
            throw new BadNetworkParameterException('Total records is empty');
        }

        $reflection = new \ReflectionFunction($read_records_callback);
        $parameters = $reflection->getParameters();
        if (count($parameters) != 4) {
            throw new BadNetworkParameterException('Callback function should have 4 parameters, func($page, $total_per_page, &$input_data, &$expected_output)');
        }

        $tmp_input_file = tempnam(sys_get_temp_dir(), 'nn_training_data');
        $tmp_input_file_handle = fopen($tmp_input_file, 'w+');
        if ($tmp_input_file_handle === false) {
            throw new CriticalTrainingErrorException('Could not create a temporary file for training data');
        }

        $tmp_output_file = tempnam(sys_get_temp_dir(), 'nn_training_data');
        $tmp_output_file_handle = fopen($tmp_output_file, 'w+');
        if ($tmp_output_file_handle === false) {
            throw new CriticalTrainingErrorException('Could not create a temporary file for training data');
        }

        $total_pages = ceil($total_records / $total_per_page);
        $last_page_count = ($total_records % $total_per_page) === 0 ? $total_per_page : ($total_records % $total_per_page);

        for ($page = 0; $page < $total_pages; $page++) {
            $input_data = [];
            $expected_output = [];
            $read_records_callback($page, $total_per_page, $input_data, $expected_output);

            if (count($input_data) == 0) {
                throw new BadNetworkParameterException('Input data is empty');
            }
            if (count($input_data) > $total_per_page) {
                throw new BadNetworkParameterException('Input data is more than the total per page');
            }
            if (count($input_data[0]) !== $this->total_input_neurons) {
                throw new BadNetworkParameterException('Input data is not the same size as the expected input, if theirs less values than usual then pad with zero');
            }
            if (count($expected_output) == 0) {
                throw new BadNetworkParameterException('Expected output data is empty');
            }
            if (count($expected_output) !== count($input_data)) {
                throw new BadNetworkParameterException('Expected output data is not the same size as the input data');
            }
            if (count($expected_output[0]) !== $this->total_output_neurons) {
                throw new BadNetworkParameterException('Expected output data is not the same size as the expected output, if theirs less values than usual then pad with zero');
            }

            foreach ($input_data as $input) {
                fwrite($tmp_input_file_handle, pack('d*', ...$input));
            }

            foreach ($expected_output as $output) {
                fwrite($tmp_output_file_handle, pack('d*', ...$output));
            }
        }

        fseek($tmp_input_file_handle, 0);
        fseek($tmp_output_file_handle, 0);

        for ($i = 0; $i < $this->epochs; $i++) {
            for ($b = 0; $b < $total_pages; $b++) {
                $input_data_arr = [];
                $output_data_arr = [];

                $total_in_page = $total_per_page;
                if ($b == $total_pages - 1) {
                    $total_in_page = $last_page_count;
                }

                for ($j = 0; $j < $total_in_page; $j++) {
                    $input_data = unpack('d*', fread($tmp_input_file_handle, $this->total_input_neurons * 8));
                    $input_data_arr[$j] = array_values($input_data);

                    $output_data = unpack('d*', fread($tmp_output_file_handle, $this->total_output_neurons * 8));
                    $output_data_arr[$j] = array_values($output_data);
                }

                $this->trainNetworkEpoch($input_data_arr, $output_data_arr);
            }

            fseek($tmp_input_file_handle, 0);
            fseek($tmp_output_file_handle, 0);
        }

        fclose($tmp_input_file_handle);
        unlink($tmp_input_file);

        fclose($tmp_output_file_handle);
        unlink($tmp_output_file);
    }

    public function saveNetwork(string $filename) {
        $data = [
            'total_input_neurons' => $this->total_input_neurons,
            'total_output_neurons' => $this->total_output_neurons,
            'learning_rate' => $this->learningRate,
            'epochs' => $this->epochs,
            'total_hidden_layer_neurons' => $this->total_hidden_layer_neurons,
            'total_hidden_layers' => $this->total_hidden_layers,
            'activation_function' => $this->activation_function,
            'layers' => []
        ];

        foreach ($this->layers as $layer) {
            $layer_data = [
                'neurons' => []
            ];

            foreach ($layer->neurons as $neuron) {
                $neuron_data = [
                    'weights' => $neuron->weights,
                    'output' => $neuron->output,
                    'delta' => $neuron->delta
                ];

                $layer_data['neurons'][] = $neuron_data;
            }

            $data['layers'][] = $layer_data;
        }

        file_put_contents($filename, json_encode($data));
    }

    public static function loadNetwork(string $filename) : NeuralNetwork
    {
        $data = json_decode(file_get_contents($filename), true);
        $network = new NeuralNetwork(
            $data['total_input_neurons'],
            $data['total_output_neurons'],
            $data['learning_rate'],
            $data['epochs'],
            $data['total_hidden_layer_neurons'],
            $data['total_hidden_layers'],
            $data['activation_function']
        );
    
        // Clear the initially created layers
        $network->layers = [];
    
        foreach ($data['layers'] as $layer_data) {
            $neuron_count = count($layer_data['neurons']);
            $prev_neuron_count = count($network->layers) == 0 ? $data['total_input_neurons'] : count($network->layers[count($network->layers) - 1]->neurons);
            
            $layer = new Layer($neuron_count, $prev_neuron_count);
            $layer->neurons = [];
    
            foreach ($layer_data['neurons'] as $neuron_data) {
                $neuron = new Neuron($prev_neuron_count);
                $neuron->weights = $neuron_data['weights'];
                $neuron->output = $neuron_data['output'];
                $neuron->delta = $neuron_data['delta'];
    
                $layer->neurons[] = $neuron;
            }
    
            $network->layers[] = $layer;
        }
    
        return $network;
    }
    
}
