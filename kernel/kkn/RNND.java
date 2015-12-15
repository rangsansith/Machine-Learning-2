package hw7.kernel.kkn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

public class RNND {

	static int featlen=200;
	static int classno=10;
	static int trainlen=5000,testlen=10000;
	static int codesize=50;
	static double [] labelstrain = new double[trainlen];
	static double [] labelstest= new double[testlen];
	static double[][] featurestrain = new double[trainlen][featlen];
	static double[][] featurestest = new double[testlen][featlen];
	static int kerneltype=1;
	static double sigma=100;	
	static double c=2;
	static double b=0.10;
	static double a=10.0;
	static int k=7;
	static double R=-0.75;
	private static void readData()
	{
		try {
			FileReader featureread = new FileReader("data/digitsdataset/trainecoc.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					featurestrain[ind][i]=Double.parseDouble(feats[i]);	
				}
				labelstrain[ind]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/digitsdataset/testecoc.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length-1;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);	
				}
				labelstest[ind]=Double.parseDouble(feats[feats.length-1]);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	private static double dotproduct(double[] i,double[] j){
		double ans=0.0;
		for(int k=0;k<i.length;k++)
		{
			ans+=i[k]*j[k];
		}
		return ans;
	}
	private static double euclidist(double[] i, double[] j) {
		double sum=0;
		for (int k=0;k<i.length;k++) {
			sum+=(i[k]-j[k])*(i[k]-j[k]);
		}
		return sum;
	}
	private static double cosine(double[] i, double[] j) {
		return dotproduct(i, j)/(Math.sqrt(dotproduct(j, j))*Math.sqrt(dotproduct(i, i)));
	}
	private static double kernel(double[] i, double[] j){
		if(kerneltype==1)
		{
			return -Math.abs(cosine(i, j));
		}
		else if(kerneltype==2)
		{
			return -Math.exp(-(euclidist(i, j)/(2*sigma)));
		}
		else if(kerneltype==3)
		{
			return -Math.pow(a*dotproduct(i, j)+b, c);
		}
		else if(kerneltype==4)
		{
			return Math.sqrt(euclidist(i, j));
		}
		else
		{
			return dotproduct(i, j);
		}
	}
	private static Double[] getMean(double[][] features,int x,int y){
		Double[] mean=new Double[y];
		for (int i = 0; i < y; i++) {
			mean[i]=0.0;
			double cnt=0;
			for (int j = 0; j < x; j++) {
				if(!Double.isNaN(features[j][i]))
				{
					mean[i]+=features[j][i];
					cnt++;
				}
			}
			mean[i]/=cnt;
		}
		return mean;
	}
	private static Double[] getStd(double[][] features,int x,int y,Double[] mean){
		Double[] std=new Double[y];
		for (int i = 0; i < y; i++) {
			std[i]=0.0;
			double cnt=0;
			for (int j = 0; j < x; j++) {
				if(!Double.isNaN(features[j][i]))
				{
					std[i]+=Math.pow(features[j][i]-mean[i],2);
					cnt++;
				}
			}
			std[i]=Math.sqrt(std[i]/cnt);
		}
		return std;
	}
	private static void normalize(double[][] features,int x,int y,Double[] mean,Double[] std){
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				if(!Double.isNaN(features[j][i]))
				{
					features[j][i]-=mean[i];
					features[j][i]/=std[i];
				}
			}
		}
	}
	private static void noramalizeboth(double[][] features,double[][] featurestest,int trx,int tex,int y){
		Double[] mean=getMean(features, trx, y);
		Double[] std=getStd(features, trx, y, mean);
		normalize(features, trx, y, mean, std);
		normalize(featurestest, tex, y, mean, std);
	}
	public static void main(String[] args) {
		readData();
		noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		double acc=0.0,dacc=0.0,tacc=0.0;
		for(int i=0;i<testlen;i++)
		{
			TreeMap<Double, ArrayList<Integer>> topk = new TreeMap<Double, ArrayList<Integer>>();
			for (int j = 0; j < trainlen; j++) {
				double dist=kernel(featurestest[i], featurestrain[j]);
				ArrayList<Integer> temp = new ArrayList<Integer>();
				if(topk.containsKey(dist))
					temp=topk.get(dist);
				temp.add(j);
				topk.put(dist, temp);
			}
			int trun=0;
			int tpredict[]=new int[10];
			for (int j = 0; j < tpredict.length; j++) {
				tpredict[j]=0;
			}
			for (Double key:topk.keySet()) {
				if(key>R)
					break;
				ArrayList<Integer> temp = topk.get(key);
				for (int j = 0; j < temp.size(); j++) {
					tpredict[(int) labelstrain[temp.get(j)]]++;
					//System.out.println("inga "+key+" "+labelstrain[temp.get(j)]);
					//System.out.println("find "+euclidist(featurestest[i], featurestrain[temp.get(j)]));
					/*trun++;
					if(trun==k)
						break;*/
				}	
				/*if(trun==k)
					break;*/
			}
			int maxcnt=0;int ind=-1;
			for (int j = 0; j < tpredict.length; j++) {
				if(maxcnt<tpredict[j])
				{
					maxcnt=tpredict[j];
					ind=j;
				}
			}
			//System.out.println(ind+" "+labelstest[i]);
			if(ind==labelstest[i])
				tacc++;
			System.out.println("Acc method 3: "+tacc/(double) (i+1));
		}
		System.out.println("Acc method 3: "+tacc/(double) testlen);

	}

}
