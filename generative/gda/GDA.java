package hw3.generative.gda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;

public class GDA {

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
	static double [] labelstest= new double[1];
	static Double[] globalmean;
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
	private static Double[] getMean(double[][] features,int x,int y){
		Double[] mean=new Double[y];
		for (int i = 0; i < y; i++) {
			mean[i]=0.0;
			for (int j = 0; j < x; j++) {
				mean[i]+=features[j][i];
			}
			mean[i]/=x;
		}
		return mean;
	}
	private static Double[] getStd(double[][] features,int x,int y,Double[] mean){
		Double[] std=new Double[y];
		for (int i = 0; i < y; i++) {
			std[i]=0.0;
			for (int j = 0; j < x; j++) {
				std[i]+=Math.pow(features[j][i]-mean[i],2);
			}
			std[i]=Math.sqrt(std[i]/x);
		}
		return std;
	}
	private static double[][] getMeanbyClass(double[][] features,double[] labels,int x,int y,double[] val){
		double[][] mean=new double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				mean[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						mean[k][i]+=features[j][i];
						len[k]++;
					}
				}
				mean[k][i]/=len[k];
			}
		}
		return mean;
	}
	private static void normalize(double[][] features,int x,int y,Double[] mean,Double[] std){
		for (int i = 0; i < y; i++) {
			for (int j = 0; j < x; j++) {
				features[j][i]-=mean[i];
				features[j][i]/=std[i];
			}
		}
	}
	private static void noramalizeboth(double[][] features,double[][] featurestest,int trx,int tex,int y){
		Double[] mean=getMean(features, trx, y);
		Double[] std=getStd(features, trx, y, mean);
		normalize(features, trx, y, mean, std);
		normalize(featurestest, tex, y, mean, std);
	}
	private static Matrix getMean(double[][] x)
	{
		int row=x.length,col=x[0].length;
		double[] mean =new double[col];
		for (int i = 0; i < col; i++) {
			mean[i]=0;
			for (int j = 0; j < row; j++) {
				mean[i]+=x[j][i];
			}
			mean[i]/=(double) row;
		}
		return new Matrix(mean,1);
	}
	private static Matrix getCovariance(double[][] x,Matrix mu){
		Matrix covariance=new Matrix(mu.getColumnDimension(), mu.getColumnDimension());
		for(int i=0;i<x.length;i++)
		{
			Matrix xtemp=new Matrix(x[i],1);
			Matrix xmu=xtemp.minus(mu);
			Matrix temp=(xmu.transpose()).times(xmu);
			covariance.plusEquals(temp);
		}
		covariance.timesEquals(1/((double) x.length));
		return covariance;
	}
	private static void dispMAtrix(Matrix a){
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				System.out.print(a.get(i, j)+"  ");
			}
			System.out.println();
		}
	}
	private static double CalculateGDA(Matrix x, Matrix mu, Matrix icovariance,double det,double n) {
		Matrix xmmu= x.minus(mu);
		Matrix xmmut= xmmu.transpose();
		Matrix total=(xmmu.times(icovariance)).times(xmmut);
		return Math.exp(-0.5*total.det())/(Math.pow(Math.PI, n/2.0)*Math.sqrt(det));
	}
	private static double evaluateGDA(double[][]featureseval,double[] labelseval,Matrix[] icovariance,Matrix[] mumat,double[] covdet,double spam,double ham,int tetrlen,double threshold)
	{
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		double spamprob=1.0,hamprob=1.0,acc=0.0;
		for (int i = 0; i < tetrlen; i++) {
			spamprob=spam;
			hamprob=ham;
			Matrix x= new Matrix(featureseval[i],1);
			spamprob*=CalculateGDA(x,mumat[1],icovariance[1],covdet[1],featlen);
			hamprob*=CalculateGDA(x,mumat[0],icovariance[0],covdet[0],featlen);
			if(spamprob>hamprob)
			{
				if(labelseval[i]==1)
				{
					acc++;
					tp++;
				}
				else
				{
					fp++;
				}
			}
			else
			{
				if(labelseval[i]==0)
				{
					acc++;
					tn++;
				}
				else
				{
					fn++;
				}
			}
		}
		acc/=((double) tetrlen);
		System.out.println("Accuracy: "+acc*100+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;

	}
	private static double[] GDAsep()
	{
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
				// estimate phi
				double classval[]=new double[classno];
				double classcount[]=new double[classno];
				classval[0]=0;
				classval[1]=1;
				double phi=0.0;
				for (int i = 0; i < trainlen; i++) {
					phi+=labelstrain[i];
				}
				classcount[0]=1-phi;
				classcount[1]=phi;
				phi/=(double) trainlen;
				// p(y)
				double[] y={1.0-phi,phi};
				// finding Mu1 and Mu2
				double[][]mu = getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, classval);
				//convert to Matrices
				Matrix[] mumat= new Matrix[classno];
				for (int i = 0; i < classno; i++) {
					mumat[i]=new Matrix(mu[i], 1);
				}
				// finding cov
				Matrix covariance[]=new Matrix[classno];
				covariance[0]=new Matrix(featlen,featlen);
				covariance[1]=new Matrix(featlen,featlen);
				for (int j = 0; j < trainlen; j++) {
					Matrix x=new Matrix(featurestrain[j],1);
					x.minusEquals(mumat[1]);
					Matrix temp=x.transpose().times(x);
					covariance[1].plusEquals(temp);
					x=new Matrix(featurestrain[j],1);
					x.minusEquals(mumat[0]);
					temp=x.transpose().times(x);
					covariance[0].plusEquals(temp);
				}
				covariance[0].timesEquals(1.0/(double) trainlen);
				covariance[1].timesEquals(1.0/(double) trainlen);
				// Determinant of covariance
				double covdet[]=new double[classno];
				covdet[0]=covariance[0].det();
				covdet[1]=covariance[1].det();
				//inverse of covariance
				Matrix[] icovariance=new Matrix[classno];
				icovariance[0]=covariance[0].inverse();
				icovariance[1]=covariance[1].inverse();
				//evaluation
				double[] acc=new double[2];
				System.out.println("Training");
				acc[0]=evaluateGDA(featurestrain, labelstrain, icovariance, mumat, covdet, y[1], y[0], trainlen, 0);
				System.out.println("Testing");
				acc[1]=evaluateGDA(featurestest, labelstest, icovariance, mumat, covdet, y[1], y[0], testlen, 0);
				return acc;
	}
	
	private static double[] GDAcom()
	{
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
				// estimate phi
				double classval[]=new double[classno];
				double classcount[]=new double[classno];
				classval[0]=0;
				classval[1]=1;
				double phi=0.0;
				for (int i = 0; i < trainlen; i++) {
					phi+=labelstrain[i];
				}
				classcount[0]=1-phi;
				classcount[1]=phi;
				phi/=(double) trainlen;
				// p(y)
				double[] y={1.0-phi,phi};
				// finding Mu1 and Mu2
				double[][]mu = getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, classval);
				//convert to Matrices
				Matrix[] mumat= new Matrix[classno];
				for (int i = 0; i < classno; i++) {
					mumat[i]=new Matrix(mu[i], 1);
				}
				// finding cov
				Matrix covariance[]=new Matrix[classno];
				covariance[0]=new Matrix(featlen,featlen);
				covariance[1]=new Matrix(featlen,featlen);
				for (int j = 0; j < trainlen; j++) {
					Matrix x=new Matrix(featurestrain[j],1);
					x.minusEquals(mumat[(int) labelstrain[j]]);
					Matrix temp=x.transpose().times(x);
					covariance[0].plusEquals(temp);
					covariance[1].plusEquals(temp);
				}
				covariance[0].timesEquals(1.0/(double) trainlen);
				covariance[1].timesEquals(1.0/(double) trainlen);
				// Determinant of covariance
				double covdet[]=new double[classno];
				covdet[0]=covariance[0].det();
				covdet[1]=covariance[1].det();
				//inverse of covariance
				Matrix[] icovariance=new Matrix[classno];
				icovariance[0]=covariance[0].inverse();
				icovariance[1]=covariance[1].inverse();
				//evaluation
				double[] acc=new double[2];
				System.out.println("Training");
				acc[0]=evaluateGDA(featurestrain, labelstrain, icovariance, mumat, covdet, y[1], y[0], trainlen, 0);
				System.out.println("Testing");
				acc[1]=evaluateGDA(featurestest, labelstest, icovariance, mumat, covdet, y[1], y[0], testlen, 0);
				return acc;
	}
	public static void main(String[] args) {

		readData();
		//randSplits(9);
		int fold=10;
		uniformSplits(fold);
		double[] accuracy={0.0,0.0};
		for (int i = 1; i <= fold; i++) {
			System.out.println("Fold:"+i);
			pickKfold(i);
			double[] temp = GDAsep();
			accuracy[0]+=temp[0];
			accuracy[1]+=temp[1];
			System.out.println();
		}	
		accuracy[0]/=(double) fold;
		accuracy[1]/=(double) fold;
		System.out.println("Overall:");
		System.out.println("Training Accuracy: "+accuracy[0]*100);
		System.out.println("Testing Accuracy: "+accuracy[1]*100);
	}

}
