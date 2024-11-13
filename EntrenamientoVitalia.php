
<?php
$serverName = "ACER-NATIS";
$connectionInfo = array("Database"=>"hospital", "UID"=>"NATALIA", "PWD"=>"root");
$conn = sqlsrv_connect($serverName, $connectionInfo);

if ($conn) {
    $sql = "SELECT * FROM ENTRENAMIENTO_VITALIA";
    $stmt = sqlsrv_query($conn, $sql);
    
    $rows = array();

    while ($row = sqlsrv_fetch_array($stmt, SQLSRV_FETCH_ASSOC)) {
        $rows[] = $row;  // Agregar cada fila al array
    }
    // Convertir el array en JSON
    $json_data = json_encode($rows);

    // Establecer el encabezado para indicar que es JSON
    header('Content-Type: application/json');

    // Imprimir el JSON (se puede guardar en un archivo o devolver en una API)
    echo $json_data;

    // Cerrar la conexión
    sqlsrv_free_stmt($stmt);
    sqlsrv_close($conn);

}
?>
