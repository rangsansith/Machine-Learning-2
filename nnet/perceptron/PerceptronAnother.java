package hw2.nnet.perceptron;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class PerceptronAnother {

	public static void main(String[] args) {
		double[][] features = new double[1001][5];
		double [][] labels = new double[1001][1];
		//HashMap<Double[], Double> featlabel= new HashMap<Double[], Double>();
		double[] mean= new double[1000];
		double[] std= new double[1000];
		try {
			FileReader featureread = new FileReader("data/perceptronData.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split("	");
				double mul=1.0;
				labels[ind][0]=Double.parseDouble(feats[feats.length-1]);
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=mul*Double.parseDouble(feats[i]);
					mean[i]+=features[ind][i];
				}
				features[ind][feats.length-1]=1.0;
				//labels[ind][0]*=mul;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		int iter=1,tmis=10;
		double[] w=new double[5];
		for (int i = 0; i < w.length; i++) {
			w[i]=0;
		}
		/*for (int i = 0; i < features.length; i++) {
			for (int j = 0; j < w.length; j++) {
				System.out.print(features[i][j]+" ");
			}
			System.out.println(labels[i][0]);
		}*/
		ArrayList<Integer> misclass = new ArrayList<Integer>();
		ArrayList<Double> misclasserror = new ArrayList<Double>();
		double hxod=0.0,hx=10.0,lambda=0.05;
		while(tmis!=0)
		{
			hx=0.0;
			for (int i = 0; i < features.length; i++) {
				hxod=0.0;
				for (int j = 0; j < w.length; j++) {
					hxod+=w[j]*features[i][j];
				}
				if(hxod<=0)
				{
					if(labels[i][0]==1)
					{
						misclass.add(i);
						misclasserror.add(1.0);
						hx++;
					}
				}
				else
				{
					if(labels[i][0]==-1)
					{
						misclass.add(i);
						misclasserror.add(-1.0);
						hx++;	 	
					}
				}
			}
			for (int i = 0; i < misclass.size(); i++) {
				for (int j = 0; j < w.length; j++) {
					w[j]=w[j]+lambda*misclasserror.get(i)*features[misclass.get(i)][j];
				}	
			}
			System.out.println("Iteration "+iter+" ,total_mistake "+misclass.size()+"  "+hx/1000.0);
			tmis=misclass.size();
			misclass = new ArrayList<Integer>();
			misclasserror= new ArrayList<Double>();
			/*try {
				System.in.read();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}*/
			iter++;
			
		}
		for (int i = 0; i < w.length; i++) {
			System.out.println(w[i]);
		}
		System.out.println("normalized");
		for (int i = 0; i < w.length-1; i++) {
			System.out.println(w[i]/w[4]);
		}

	}

}
