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
class FPRComparator implements Comparator<Double[]> {
	public int compare(Double o1[], Double o2[]) {
		if(o1[1].compareTo(o2[1])==0)
			return o1[0].compareTo(o2[0]);
		else
			return o1[1].compareTo(o2[1]);
	}
}
@SuppressWarnings("serial")
public class RocAuc extends JFrame{
	public static HashMap<Integer, ArrayList<Double[]>> alldata = new HashMap<Integer, ArrayList<Double[]>>();
	public static HashMap<Integer, ArrayList<Double[]>> allerror = new HashMap<Integer, ArrayList<Double[]>>();
	public static ArrayList<Double[]> allauc = new ArrayList<Double[]>();
	public static ArrayList<Double[]> allre = new ArrayList<Double[]>();
	public RocAuc(int a) {
		//super("ROC for spam dataset");
       if(a==1){
		JPanel chartPanel = createChartPanel();
		add(chartPanel, BorderLayout.CENTER);
		setSize(640, 480);
		//setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
       }
       else if(a==2){
    	JPanel chartPanel = createChartPanelerror();
   		add(chartPanel, BorderLayout.CENTER);
   		setSize(640, 480);
   		//(JFrame.EXIT_ON_CLOSE);
   		setLocationRelativeTo(null);   
       }
       else if(a==3){
       	JPanel chartPanel = createChartPanelrerror();
      		add(chartPanel, BorderLayout.CENTER);
      		setSize(640, 480);
      		//setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      		setLocationRelativeTo(null);   
          }
       else {
       	JPanel chartPanel = createChartPanelauc();
      		add(chartPanel, BorderLayout.CENTER);
      		setSize(640, 480);
      		//setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      		setLocationRelativeTo(null);   
          }
	}
	private JPanel createChartPanel() {
		String chartTitle = "ROC for spam dataset";
		String xAxisLabel = "FPR(False Positive Rate)";
		String yAxisLabel = "TPR(True Positive Rate)";

		XYDataset dataset = createDataset();

		JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
				xAxisLabel, yAxisLabel, dataset);

		return new ChartPanel(chart);
	}
	private XYDataset createDataset() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries lr = new XYSeries("ROC");
		XYSeries ref = new XYSeries("Reference");
		XYSeries all[]={lr,ref};
		for (int i=0;i<alldata.size();i++) {
			for (int j = 0; j < alldata.get(i).size(); j++) {
				all[i].add(alldata.get(i).get(j)[1], alldata.get(i).get(j)[0]);	
			}
			dataset.addSeries(all[i]);
		}
		return dataset;
	}
	private JPanel createChartPanelerror() {
		String chartTitle = "Train and Test error for spam dataset";
		String xAxisLabel = "Iteration";
		String yAxisLabel = "Error";

		XYDataset dataset = createDataseterror();

		JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
				xAxisLabel, yAxisLabel, dataset);

		return new ChartPanel(chart);
	}
	private XYDataset createDataseterror() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries tr = new XYSeries("Training");
		XYSeries te = new XYSeries("Testing");
		XYSeries all[]={tr,te};
		for (int i=0;i<allerror.size();i++) {
			for (int j = 0; j < allerror.get(i).size(); j++) {
				all[i].add(allerror.get(i).get(j)[1], allerror.get(i).get(j)[0]);	
			}
			dataset.addSeries(all[i]);
		}
		return dataset;
	}
	private JPanel createChartPanelrerror() {
		String chartTitle = "Rounderror error for spam dataset";
		String xAxisLabel = "Iteration";
		String yAxisLabel = "Round Error";

		XYDataset dataset = createDatasetrerror();

		JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
				xAxisLabel, yAxisLabel, dataset);

		return new ChartPanel(chart);
	}
	private XYDataset createDatasetrerror() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries tr = new XYSeries("ROUND ERROR");
			for (int j = 0; j < allre.size(); j++) {
				tr.add(allre.get(j)[1], allre.get(j)[0]);	
			}
			dataset.addSeries(tr);
		return dataset;
	}
	private JPanel createChartPanelauc() {
		String chartTitle = "Auc for spam dataset";
		String xAxisLabel = "Iteration";
		String yAxisLabel = "AUC";

		XYDataset dataset = createDatasetauc();

		JFreeChart chart = ChartFactory.createXYLineChart(chartTitle,
				xAxisLabel, yAxisLabel, dataset);

		return new ChartPanel(chart);
	}
	private XYDataset createDatasetauc() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		XYSeries tr = new XYSeries("AUC");
			for (int j = 0; j < allauc.size(); j++) {
				tr.add(allauc.get(j)[1], allauc.get(j)[0]);	
			}
			dataset.addSeries(tr);
		return dataset;
	}

	private static double CalculateAUC(ArrayList<Double[]> temp) {
		Collections.sort(temp,new FPRComparator());
		double auc=0.0;
		for (int i = 1; i < temp.size(); i++) {
			auc+=0.5*(temp.get(i)[0]+temp.get(i-1)[0])*(temp.get(i)[1]-temp.get(i-1)[1]);	
		}
		return auc;
	}

	public static void main(String[] args) {

		try {
			//HashMap<Double, Double> tempuniq= new HashMap<Double, Double>();
			FileReader tmp = new FileReader("resulthw4/auc");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			int iter=1;
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				Double[] tempd={Double.parseDouble(sCurrentLine),(double) iter};
				allauc.add(tempd);
				iter++;
			}
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			//HashMap<Double, Double> tempuniq= new HashMap<Double, Double>();
			FileReader tmp = new FileReader("resulthw4/rounderror");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			int iter=1;
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				Double[] tempd={Double.parseDouble(sCurrentLine),(double) iter};
				allre.add(tempd);
				iter++;
			}
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			//HashMap<Double, Double> tempuniq= new HashMap<Double, Double>();
			FileReader tmp = new FileReader("resulthw4/trainerror");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			int iter=1;
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				Double[] tempd={Double.parseDouble(sCurrentLine),(double) iter};
				temp.add(tempd);
				iter++;
			}
			allerror.put(0, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			//HashMap<Double, Double> tempuniq= new HashMap<Double, Double>();
			FileReader tmp = new FileReader("resulthw4/testerror");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			int iter=1;
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				Double[] tempd={Double.parseDouble(sCurrentLine),(double) iter};
				temp.add(tempd);
				iter++;
			}
			allerror.put(1, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		double auc=0.0;
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			//HashMap<Double, Double> tempuniq= new HashMap<Double, Double>();
			FileReader tmp = new FileReader("resulthw4/roc");
			BufferedReader tmpbr = new BufferedReader(tmp);
			String sCurrentLine;
			String[] cc;
			while ((sCurrentLine = tmpbr.readLine()) != null) {
				cc=sCurrentLine.split(" ");
				Double tp=Double.parseDouble(cc[1]),fp=Double.parseDouble(cc[2]),tn=Double.parseDouble(cc[3]),fn=Double.parseDouble(cc[4]);
				Double tprfpr[]={tp/(tp+fn),fp/(fp+tn)};
				if(tp==0)
					tprfpr[0]=0.0;
				else
					tprfpr[0]=tp/(tp+fn);
				if(fp==0)
					tprfpr[1]=0.0;
				else
					tprfpr[1]=fp/(fp+tn);
				temp.add(Integer.parseInt(cc[0]),tprfpr);
			}
			auc=CalculateAUC(temp);
			alldata.put(0, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println("Auc: "+auc);
		ArrayList<Double[]> temp = new ArrayList<Double[]>();
		double val=0.0,step=1.0/920.0;
		for (int i = 0; i < 921; i++) {
			Double t[]={val,val};
			temp.add(i,t);
			val+=step;
		}
		alldata.put(1, temp);
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				new RocAuc(1).setVisible(true);
				new RocAuc(2).setVisible(true);
				new RocAuc(3).setVisible(true);
				new RocAuc(4).setVisible(true);
			}
		});
	}
}
