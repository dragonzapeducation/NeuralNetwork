<?php
namespace Dragonzap\NeuralNetwork;

class Neuron
{
    public $weights = [];
    public $output = 0;
    public $delta = 0;

    public function __construct($total_last_layer_neurons)
    {
        for ($i = 0; $i < $total_last_layer_neurons; $i++) {
            $this->weights[] = (double) rand() / getrandmax();
        }
    }
}
