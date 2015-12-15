package hw2.report.roc;

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
        return o1[1].compareTo(o2[1]);
    }
}
@SuppressWarnings("serial")
public class RocAuc extends JFrame{
	public static HashMap<Integer, ArrayList<Double[]>> alldata = new HashMap<Integer, ArrayList<Double[]>>();
	public RocAuc() {
		super("ROC for spam dataset");

		JPanel chartPanel = createChartPanel();
		add(chartPanel, BorderLayout.CENTER);

		setSize(640, 480);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
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
		XYSeries lr = new XYSeries("Linear Regression");
		XYSeries lrr = new XYSeries("Ridge Regression");
		XYSeries lgr = new XYSeries("Gradient Regression");
		XYSeries logr = new XYSeries("Logistic Regression");
		XYSeries ref = new XYSeries("Reference");
		XYSeries all[]={lr,lrr,lgr,logr,ref};
		for (int i=0;i<alldata.size();i++) {
			for (int j = 0; j < alldata.get(i).size(); j++) {
				all[i].add(alldata.get(i).get(j)[1], alldata.get(i).get(j)[0]);	
			}
			dataset.addSeries(all[i]);
		}
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

		double auc=0.0;
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("result/spam_cclr");
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
		System.out.println("Regression Auc: "+auc);
		auc=0.0;
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("result/spam_cclrr");
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
			alldata.put(1, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println("Ridge Regression Auc: "+auc);
		auc=0.0;
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("result/spam_cclgr");
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
			alldata.put(2, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println("Gradient Auc: "+auc);
		auc=0.0;
		try {
			ArrayList<Double[]> temp = new ArrayList<Double[]>();
			FileReader tmp = new FileReader("result/spam_cclogr");
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
			alldata.put(3, temp);
			tmpbr.close();
			tmp.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		System.out.println("Logistic Auc: "+auc);
		ArrayList<Double[]> temp = new ArrayList<Double[]>();
		double val=0.0,step=1.0/920.0;
		for (int i = 0; i < 921; i++) {
			Double t[]={val,val};
			temp.add(i,t);
			val+=step;
		}
		alldata.put(4, temp);
		SwingUtilities.invokeLater(new Runnable() {
			@Override
			public void run() {
				new RocAuc().setVisible(true);
			}
		});
	}
}
