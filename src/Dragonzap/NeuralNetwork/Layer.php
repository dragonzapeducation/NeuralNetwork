<?php

namespace Dragonzap\NeuralNetwork;

class Layer
{
    public $neurons = [];
    public $output = [];

    public function __construct($total_neurons, $total_last_layer_neurons)
    {
        for ($i = 0; $i < $total_neurons; $i++) {
            $neuron = new Neuron($total_last_layer_neurons);
            $this->neurons[] = $neuron;
            $this->output[] = 0;
        }
    }
}