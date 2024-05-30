<?php
namespace Dragonzap\Exceptions;
use Exception;
class BadNetworkParameterException extends Exception
{
    public function __construct($message = "Bad network parameter", $code = 0, Exception $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}