<?php
$serverName="ACER-NATIS";
$connectionInfo=array("Database"=>"hospital","UID"=>"NATALIA","PWD"=>"root");
$conn=sqlsrv_connect($serverName,$connectionInfo);

if($conn){
	
}else{
	echo"Conexion no se pudo establecer.<br/>";
	die(print_r(sqlsrv_errors(),true));
}
?>