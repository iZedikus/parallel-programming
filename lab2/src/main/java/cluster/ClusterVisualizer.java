package cluster;

import javafx.application.Application;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Tab;
import javafx.scene.control.TabPane;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class ClusterVisualizer extends Application {
    private static volatile List<List<CentroidCluster<DoublePoint>>> allClusters = new ArrayList<>();
    private static volatile List<RealMatrix> allData = new ArrayList<>();
    private static volatile List<Integer> allK = new ArrayList<>();

    @Override
    public void start(Stage primaryStage) {
        TabPane tabPane = new TabPane();

        for (int i = 0; i < allK.size(); i++) {
            Tab tab = new Tab("k=" + allK.get(i));
            tab.setContent(createScatterChart(allClusters.get(i), allData.get(i), allK.get(i)));
            tab.setClosable(true);
            tabPane.getTabs().add(tab);
        }

        VBox root = new VBox(tabPane);
        Scene scene = new Scene(root, 600, 600);
        primaryStage.setScene(scene);
        primaryStage.setTitle("Кластерный анализ");
        primaryStage.show();
    }

    public static void addData(List<CentroidCluster<DoublePoint>> clusters, RealMatrix data, int k) {
        allClusters.add(clusters);
        allData.add(data);
        allK.add(k);
    }

    public static void visualize() {
        launch();
    }

    private Node createScatterChart(List<CentroidCluster<DoublePoint>> clusters,
                                    RealMatrix data, int k) {
        // Оси
        NumberAxis xAxis = new NumberAxis(-0.01, 1.01, 0.1);
        xAxis.setLabel("Creatinine (нормализованный)");
        NumberAxis yAxis = new NumberAxis(-0.01, 1.01, 0.1);
        yAxis.setLabel("HCO3 (нормализованный)");

        ScatterChart<Number, Number> scatterChart = new ScatterChart<>(xAxis, yAxis);
        scatterChart.setTitle("Центры кластеров (k = " + k + ")");


        // Точки
        XYChart.Series<Number, Number> dataSeries = new XYChart.Series<>();
        dataSeries.setName("Данные (" + data.getRowDimension() + " точек)");

        for (int i = 0; i < data.getRowDimension(); i++) {
            dataSeries.getData().add(new XYChart.Data<>(
                    data.getEntry(i, 0),
                    data.getEntry(i, 1)
            ));
        }
        scatterChart.getData().add(dataSeries);


        // Центры
        XYChart.Series<Number, Number> centersSeries = new XYChart.Series<>();
        centersSeries.setName("Центры кластеров");

        for (CentroidCluster<DoublePoint> cluster : clusters) {
            double[] center = cluster.getCenter().getPoint();
            XYChart.Data<Number, Number> centerPoint = new XYChart.Data<>(center[0], center[1]);
            centersSeries.getData().add(centerPoint);
        }
        scatterChart.getData().add(centersSeries);

        return scatterChart;
    }
}
