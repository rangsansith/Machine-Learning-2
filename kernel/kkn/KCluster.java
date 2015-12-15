package hw7.kernel.kkn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

public class KCluster {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] features = new double[len][featlen];
	static double[] labels = new double[len];
	static ArrayList<double[][]> featklist = new ArrayList<double[][]>();
	static ArrayList<double[]> labelsklist = new ArrayList<double[]>();
	static ArrayList<Integer> indexlist = new ArrayList<Integer>();
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [][] kernelmat= new double[1][1];
	static double [] labelstest= new double[1];
	static double[] alpha;
	static double b;
	static double regc=10;
	static double numtol=0.001;
	static int maxpass=10;
	static int kerneltype=2;
	static double sigmag=1.0/featlen;	
	static double tanc=0.1;
	static double tanb=0.0;
	static int k=7;
	static int ktype[] = {1,3,7};
	private static void readData()
	{
		int one=0,zero=0;
		try {
			FileReader featureread = new FileReader("data/spambase.data");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				indexlist.add(ind);
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					features[ind][i]=Double.parseDouble(feats[i]);

				}
				labels[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labels[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
	}
	private static void randSplits(int folds)
	{
		HashMap<Integer, Boolean> seen = new HashMap<Integer, Boolean>();
		int kfold=len/folds,listno=0,one=0,zero=0;
		Random n = new Random(); 
		while(indexlist.size()!=0){
			one=0;zero=0;
			int r=Math.min(kfold, indexlist.size());
			double[][] featurestemp = new double[r][featlen];
			double [] labelstemp = new double[r];
			for (int i = 0; i < kfold; i++) {
				if(indexlist.size()==0)
					break;
				int ind=n.nextInt(indexlist.size());
				int tmpind=indexlist.get(ind);
				if(seen.containsKey(tmpind))
					System.out.println("Error: bad Coder");
				else
					seen.put(tmpind, true);
				for (int j = 0; j < featlen; j++) {
					featurestemp[i][j]=features[tmpind][j];
				}
				labelstemp[i]=labels[tmpind];
				if(labels[tmpind]==0.0)
					zero++;
				else if(labels[tmpind]==1.0)
					one++;
				else
					System.out.println("Error: bad Coder");

				indexlist.remove(ind);
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void uniformSplits(int folds){
		ArrayList<ArrayList<double[]>> tft= new ArrayList<ArrayList<double[]>>();
		ArrayList<ArrayList<Double>> tlt= new ArrayList<ArrayList<Double>>();
		for (int i = 0; i < folds; i++) {
			tft.add(new ArrayList<double[]>());
			tlt.add(new ArrayList<Double>());
		}
		for (int i = 0; i < len; i++) {
			ArrayList<double[]> temp = tft.get(i%folds);
			temp.add(features[i]);
			tft.set(i%folds, temp);
			ArrayList<Double> lemp = tlt.get(i%folds);
			lemp.add(labels[i]);
			tlt.set(i%folds,lemp);
		}
		if(tft.size()!=tlt.size())
			System.out.println("Error: BadCode");
		int one=0,zero=0,listno=0;
		for (int i = 0; i < tft.size(); i++) {
			one=0;zero=0;
			double[][] featurestemp = new double[tft.get(i).size()][featlen];
			double [] labelstemp = new double[tlt.get(i).size()];
			for (int j = 0; j < tft.get(i).size(); j++) {
				featurestemp[j]=tft.get(i).get(j);
				labelstemp[j]=tlt.get(i).get(j);
				if(labelstemp[j]==0)
					zero++;
				else
					one++;
			}
			//System.out.println(zero+"   "+one+"   "+(double) zero/(zero+one));
			featklist.add(listno,featurestemp);
			labelsklist.add(listno,labelstemp);
			listno++;
		}
	}
	private static void pickKfold(int k){
		k--;
		testlen=labelsklist.get(k).length;
		trainlen=len-testlen;
		featurestrain=new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest=new double[testlen][featlen];
		labelstest=new double[testlen];
		int ind=0;
		for (int i = 0; i < featklist.size(); i++) {
			if(i==k)
			{
				featurestest=featklist.get(k);
				labelstest=labelsklist.get(k);
				continue;
			}
			double[][] tf = featklist.get(i);
			double[] tl = labelsklist.get(i);
			for (int j = 0; j < tl.length; j++) {
				featurestrain[ind]=tf[j];
				labelstrain[ind]=tl[j];
				ind++;
			}
		}
	}
	private static double[] changeleabels(double[] labelsc){
		for (int i = 0; i < labelsc.length; i++) {
			if(labelsc[i]==0)
				labelsc[i]=-1;
		}
		return labelsc;
	}
	private static double dotproduct(double[] i,double[] j){
		double ans=0;
		for(int k=0;k<featlen;k++)
		{
			ans+=i[k]*j[k];
		}
		return ans;
	}
	private static double euclidist(double[] i, double[] j) {
		double sum=0;
		for (int k=0;k<featlen;k++) {
			sum+=(i[k]-j[k])*(i[k]-j[k]);
		}
		return sum;
	}
	private static double kernel(double[] i, double[] j){
		if(kerneltype==1)
		{
			return dotproduct(i, j);
		}
		else if(kerneltype==2)
		{
			return Math.exp(-(euclidist(i, j)*(sigmag*sigmag)));
		}
		else if(kerneltype==3)
		{
			return Math.tanh(sigmag*dotproduct(i, j)+tanb);
		}
		else
		{
			return Math.pow(sigmag*dotproduct(i, j)+tanb, tanc);
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
		int folds=10;
		uniformSplits(folds);
		for (int f = 1; f <= folds; f++) {
			System.out.println("Fold");
			for (int kt = 0; kt < ktype.length; kt++) {
				k=ktype[kt];
				System.out.println("K:"+k);
				pickKfold(f);
				noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
				labelstrain=changeleabels(labelstrain);
				labelstest=changeleabels(labelstest);
				double acc=0.0,dacc=0.0,tacc=0.0;
				for(int i=0;i<testlen;i++)
				{
					TreeMap<Double, ArrayList<Integer>> topk = new TreeMap<Double, ArrayList<Integer>>();
					TreeMap<Double, Integer> topkk = new TreeMap<Double, Integer>();

					for (int j = 0; j < trainlen; j++) {
						double dist=Math.sqrt(euclidist(featurestest[i], featurestrain[j]));
						ArrayList<Integer> temp = new ArrayList<Integer>();
						if(topk.containsKey(dist))
							temp=topk.get(dist);
						temp.add(j);
						topk.put(dist, temp);
						topkk.put(dist, j);
					}
					double predict=0.0;int run=0;
					double dpredict=0.0;int drun=0;
					double tpredict=0.0;int trun=0;
					for (Double key:topk.keySet()) {
						ArrayList<Integer> temp = topk.get(key);
						for (int j = 0; j < temp.size(); j++) {
							predict+=labelstrain[temp.get(j)];
						}	
						run++;
						if(run==k)
							break;
					}
					for (Double key:topk.keySet()) {
						ArrayList<Integer> temp = topk.get(key);
						for (int j = 0; j < temp.size(); j++) {
							tpredict+=labelstrain[temp.get(j)];
							trun++;
							if(trun==k)
								break;
						}	
						if(trun==k)
							break;
					}
					for (Double key:topkk.keySet()) {
						dpredict+=labelstrain[topkk.get(key)];
						drun++;
						if(drun==k)
							break;
					}
					if(predict*labelstest[i]>0)
						acc++;
					if(dpredict*labelstest[i]>0)
						dacc++;
					if(tpredict*labelstest[i]>0)
						tacc++;
				}
				System.out.println("Acc method 1: "+acc/(double) testlen);
				System.out.println("Acc method 2: "+dacc/(double) testlen);
				System.out.println("Acc method 3: "+tacc/(double) testlen);
			}
		}
	}

}
