package hw4.boosting.report;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
@SuppressWarnings("serial")
public class AdaPlot extends JFrame{
	public static HashMap<Integer, ArrayList<Double[]>> allerror = new HashMap<Integer, ArrayList<Double[]>>();
	public AdaPlot() {
		//super("ROC for spam dataset");
       JPanel chartPanel = createChartPanel();
		add(chartPanel, BorderLayout.CENTER);
		setSize(640, 480);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
	}
	private JPanel createChartPanel() {
		String chartTitle = "Adboost";
		String xAxisLabel = "%c";
		String yAxisLabel = "testerror";

		XYDataset dataset = createDataset();

		JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
				xAxisLabel, yAxisLabel, dataset);

		return new ChartPanel(chart);
	}
	private XYDataset createDataset() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries rand = new XYSeries("Random");
		XYSeries act = new XYSeries("Active");
		XYSeries all[]={rand,act};
		for (int i=0;i<allerror.size();i++) {
			for (int j = 0; j < allerror.get(i).size(); j++) {
				all[i].add(allerror.get(i).get(j)[0], allerror.get(i).get(j)[1]);	
			}
			dataset.addSeries(all[i]);
		}
		return dataset;
	}
	public static void main(String[] args) {

		ArrayList<Double[]> temp=new ArrayList<Double[]>();
		try {
			temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("resulthw4/randomada");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			String split[];
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				split=sCurrentLine.split(" ");
				Double[] tempd={Double.parseDouble(split[0]),Double.parseDouble(split[1])};
				temp.add(tempd);
			}
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		allerror.put(0, temp);
		try {
			temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("resulthw4/activeada");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			String split[];
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				split=sCurrentLine.split(" ");
				Double[] tempd={Double.parseDouble(split[0]),Double.parseDouble(split[1])};
				temp.add(tempd);
			}
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		allerror.put(1, temp);
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				new AdaPlot().setVisible(true);
			}
		});
	}
}
