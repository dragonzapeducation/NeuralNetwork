<?php
namespace Dragonzap\Exceptions;
use Exception;
class CriticalTrainingErrorException extends Exception
{
    public function __construct($message = "A problem with training occured.", $code = 0, Exception $previous = null)
    {
        parent::__construct($message, $code, $previous);
    }
}