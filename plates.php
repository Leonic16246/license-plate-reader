<?php
// Connect to the SQLite database
$database = new SQLite3('database/plates.db');

// Query to retrieve all plates from the database
$results = $database->query('SELECT * FROM plates');


?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>License Plate Detections</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid black;
        }
        th, td {
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        img {
            max-width: 150px; /* Limit the size of the image */
            height: auto;
        }
    </style>
</head>
<body>

<h1>Detected License Plates</h1>

<table>
    <tr>
        <th>ID</th>
        <th>Plate Number</th>
        <th>Image</th>
        <th>Confidence</th>
        <th>Timestamp</th>
    </tr>

    <?php
    // Loop through the results and display each row
    while ($row = $results->fetchArray()) {
        echo "<tr>";
        echo "<td>" . $row['id'] . "</td>";
        echo "<td>" . htmlspecialchars($row['plate_number']) . "</td>";

        // Extract only the filename from the full path
        $filename = basename($row['path_to_image']);
        echo "<td><img src='/plate_images/" . htmlspecialchars($filename) . "' alt='Plate Image'></td>";

        echo "<td>" . $row['confidence'] . "</td>";
        echo "<td>" . $row['timestamp'] . "</td>";
        echo "</tr>";
    }
    ?>

</table>

</body>
</html>