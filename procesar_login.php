<?php
session_start();
// Verifica si se envió un rol y un ID de ingreso
if(isset($_POST['rol'])) {
    $rol = $_POST['rol'];
    $IdIngreso = $_POST['IdIngreso'];
    $contrasena =$_POST['contrasena'];
    switch ($rol) {
    case 'medico':
        include("MEDICO/conexion.php");
        if (isset($_POST['IdIngreso']) && isset($_POST['contrasena'])) {
            $IdIngreso = $_POST['IdIngreso'];
            $contrasena = $_POST['contrasena'];

            // Query to retrieve the medico with the given ID
            $query = "SELECT * FROM MEDICOS WHERE ID_MED = ?";
            $params = array($IdIngreso);
            $result = sqlsrv_query($conn, $query, $params);

            if ($result && sqlsrv_has_rows($result)) {
                // Fetch the medico's data
                $medico = sqlsrv_fetch_array($result, SQLSRV_FETCH_ASSOC);
                // Check if the password matches
                if ($medico['CLAVE'] === $contrasena) {
                    $_SESSION['rol'] = "medico";
                    $_SESSION['medico'] = $medico;
                    $_SESSION['contrasena'] = $contrasena;
                    header("Location: MEDICO/medico.php");
                    exit();
                } else {
                    header("Location: log_in.html?error=wrong_password");
                    exit();
                }
            } else {
                header("Location: log_in.html?error=wrong_id");
                exit();
            }
        } else {
            // If no login data was sent, redirect back to the login form
            header("Location: log_in.html");
            exit();
        }
        break;
        case 'administrativo':
        include("ADMINISTRATIVO/conexion.php");
        if (isset($_POST['IdIngreso']) && isset($_POST['contrasena'])) {
            $IdIngreso = $_POST['IdIngreso'];
            $contrasena = $_POST['contrasena'];

            // Query to retrieve the medico with the given ID
            $query = "SELECT * FROM ADMINISTRADOR WHERE ID_ADMIN = ?";
            $params = array($IdIngreso);
            $result = sqlsrv_query($conn, $query, $params);

            if ($result && sqlsrv_has_rows($result)) {
                // Fetch the medico's data
                $admin = sqlsrv_fetch_array($result, SQLSRV_FETCH_ASSOC);
                // Check if the password matches
                if ($admin['CLAVE'] === $contrasena) {
                    $_SESSION['rol'] = "administrativo";
                    $_SESSION['administrativo'] = $admin;
                    $_SESSION['contrasena'] = $contrasena;
                    header("Location: ADMINISTRATIVO/administrativo.php");
                    exit();
                } else {
                    header("Location: log_in.html?error=wrong_password");
                    exit();
                }
            } else {
                header("Location: log_in.html?error=wrong_id");
                exit();
            }
        } else {
            // If no login data was sent, redirect back to the login form
            header("Location: log_in.html");
            exit();
        }
        break;
    }
} else {
    // Si no se enviaron datos de inicio de sesión, redirige al formulario de inicio de sesión
    echo '<script type="text/javascript">window.onload = function () { alert("el usuario no se encuentra registrado"); }</script>'; 
    echo "<script>window.location = 'log_in.html';</script>";
    exit();
}
?>