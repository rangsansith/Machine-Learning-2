package hw5.features.PCA;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import flanagan.analysis.PCA;

public class Bayes {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int pollutetrainlen=4140;
	static int pollutetestlen=461;
	static int pollutefeatlen=100;
	static int trainlen,testlen;
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [] labelstest= new double[1];
	static double[] globalmean;
	private static void readPolluted(){
		int one=0,zero=0;
		trainlen=pollutetrainlen;
		testlen=pollutetestlen;
		featlen=pollutefeatlen;
		featurestrain = new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest = new double[testlen][featlen];
		labelstest  = new double[testlen];
		try {
			FileReader featureread = new FileReader("data/spam_polluted/train100_feature.txt");
			FileReader labelsread = new FileReader("data/spam_polluted/train_label.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			BufferedReader labelsreadbr = new BufferedReader(labelsread);
			String sCurrentLine,sLabel;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sLabel = labelsreadbr.readLine();
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestrain[ind][i]=Double.parseDouble(feats[i]);

				}
				labelstrain[ind]=Double.parseDouble(sLabel);
				if(labelstrain[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			labelsreadbr.close();
			featurereadbr.close();
			labelsread.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));
		try {
			FileReader featureread = new FileReader("data/spam_polluted/test100_feature.txt");
			FileReader labelsread = new FileReader("data/spam_polluted/test_label.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			BufferedReader labelsreadbr = new BufferedReader(labelsread);
			String sCurrentLine,sLabel;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				sLabel = labelsreadbr.readLine();
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);

				}
				labelstest[ind]=Double.parseDouble(sLabel);
				if(labelstest[ind]==0.0)
					zero++;
				else
					one++;
				ind++;
			}
			labelsreadbr.close();
			featurereadbr.close();
			labelsread.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		//System.out.println(zero+"   "+one+"  "+(double) zero/(zero+one));

	}
	private static double[] getMean(double[][] features,int x,int y){
		double[] mean=new double[y];
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
	private static Double[] getStd(double[][] features,int x,int y,double[] mean){
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
	private static Double[][] getMeanbyClass(double[][] features,double[] labels,int x,int y,double[] val){
		Double[][] mean=new Double[val.length][y];
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
	private static Double[][] getCommonVarbyClass(double[][] features,double[] labels,int x,int y,double[] val, Double[][] mean){
		Double[][] std=new Double[val.length][y];
		for (int i = 0; i < y; i++) {
			std[0][i]=0.0;
			for (int j = 0; j < x; j++) 
			{
				std[0][i]+=Math.pow(features[j][i]-mean[(int) labels[j]][i],2);
			}
			std[0][i]/=(double) x;
		}
		std[1]=std[0];
		return std;
	}
	private static Double[][] getBiasedVarbyClass(double[][] features,double[] labels,int x,int y,double[] val,Double[][] mean){
		Double[][] std=new Double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				std[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						std[k][i]+=Math.pow(features[j][i],2);
						len[k]++;
					}
				}
				std[k][i]/=(len[k])-mean[k][i];
			}
		}
		return std;
	}
	private static Double[][] getVarbyClass(double[][] features,double[] labels,int x,int y,double[] val,Double[][] mean){
		Double[][] std=new Double[val.length][y];
		Double[] len=new Double[val.length];
		for (int i = 0; i < y; i++) {
			for (int k = 0; k < val.length; k++) {
				std[k][i]=0.0;
				len[k]=0.0;
				for (int j = 0; j < x; j++) {
					if(labels[j]==val[k])
					{
						std[k][i]+=Math.pow(features[j][i]-mean[k][i],2);
						len[k]++;
					}
				}
				std[k][i]/=(len[k]-1);
			}
		}
		return std;
	}
	private static void normalize(double[][] features,int x,int y,double[] mean,Double[] std){
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
		double[] mean=getMean(features, trx, y);
		Double[] std=getStd(features, trx, y, mean);
		normalize(features, trx, y, mean, std);
		normalize(featurestest, tex, y, mean, std);
	}
	private static double evaluateBernoulli(double[][]featureseval,double[] labelseval,double[][][] conditionalprob,double cc[],double spam,double ham,int tetrlen,double threshold)
	{
		double acc=0;
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		// Testing the model
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				if(!Double.isNaN(featureseval[j][k])) //same new condition
				{
					spamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][1];
					hamprob*=conditionalprob[k][(featureseval[j][k]>globalmean[k]?1:0)][0];
				}
			}
			if(Math.log(spamprob/hamprob)>threshold)
			{
				if(labelseval[j]==1)
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
				if(labelseval[j]==0)
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
		cc[0]=tp;
		cc[1]=fp;
		cc[2]=tn;
		cc[3]=fn;
		//System.out.println("Accuracy: "+acc*100+"  "+spam+"  "+ham+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;
	}
	private static double[][] Bernoulli(double[][] cc,double[] acc,boolean smoothing){
		// Estimate conditional probability
		int category=2,spamcount=0;
		double[][][] conditionalprob= new double[featlen][category][classno];
		for (int j = 0; j < featlen; j++) {
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					if(smoothing)
						conditionalprob[j][k][m]=1;  // laplace smoothing
					else
						conditionalprob[j][k][m]=0;  //no smoothing
				}
			}
			spamcount=0;
			for (int k = 0; k < trainlen; k++) {
				if(!Double.isNaN(featurestrain[k][j]))//new condition imposed train only with values that are present
					conditionalprob[j][(featurestrain[k][j]>globalmean[j]?1:0)][(int) labelstrain[k]]++;
				spamcount+=labelstrain[k];
			}
			int classlen[]=new int[2];
			if(smoothing)
			{
				classlen[0]=trainlen-spamcount+category;
				classlen[1]=spamcount+category; //laplace smoothing
			}
			else
			{ 
				classlen[0]=trainlen-spamcount;
				classlen[1]=spamcount; // no smoothing
			}
			for (int k = 0; k < category; k++) {
				for (int m = 0; m < classno; m++) {
					conditionalprob[j][k][m]/=((double) (classlen[m]));//+classno*category));
				}
			}
		}
		double spam=((double) spamcount)/((double) trainlen);
		double ham=1.0-spam;
		//evaluate
		
			acc[0]=evaluateBernoulli(featurestrain, labelstrain, conditionalprob, cc[0], spam, ham, trainlen,0.0);
			acc[1]=evaluateBernoulli(featurestest, labelstest, conditionalprob, cc[1], spam, ham, testlen,0.0);
		
		return null;
	}
	private static double GaussianModel(double mu,double var,double actval){
		double ans=1.0/(Math.sqrt(2.0*Math.PI*var));
		ans*=Math.exp(-Math.pow((actval-mu), 2)/(2.0*var));
		return ans;
	}
	private static double[][] ROCGaussian(double[][]featureseval,double[] labelseval,Double[][] classmean,Double[][] classvar,double spam,double ham,int tetrlen)
	{
		ArrayList<Double> threshlist = new ArrayList<Double>();
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=GaussianModel(classmean[1][k], classvar[1][k], featureseval[j][k]);
				hamprob*=GaussianModel(classmean[0][k], classvar[0][k], featureseval[j][k]);
			}
			if(spamprob==0)
			{
				spamprob=hamprob/10;
			}
			if(hamprob==0)
			{
				hamprob=spamprob/10;
			}
			threshlist.add(Math.log(spamprob/hamprob));
		}
		Collections.sort(threshlist);
		threshlist.add(0,threshlist.get(0)-1);
		threshlist.add(threshlist.size(),threshlist.get(threshlist.size()-1)+1);
		double maxt=0;
		double[][] thrcc=new double[threshlist.size()][4];
		for (int i = 0; i < threshlist.size(); i++) {
			maxt=Math.max(maxt,evaluateGaussian(featureseval, labelseval, classmean, classvar, thrcc[i], spam, ham, tetrlen,threshlist.get(i)));
		}
		System.out.println("Max Accuracy "+maxt);
		return thrcc;
	}
	private static double evaluateGaussian(double[][]featureseval,double[] labelseval,Double[][] classmean,Double[][] classvar,double cc[],double spam,double ham,int tetrlen,double threshold)
	{
		double acc=0;
		double tp=0.0,fp=0.0,tn=0.0,fn=0.0;
		// Testing the model
		for (int j = 0; j < tetrlen; j++) {
			double spamprob=spam,hamprob=ham;
			for (int k = 0; k < featlen; k++) {
				spamprob*=GaussianModel(classmean[1][k], classvar[1][k], featureseval[j][k]);
				hamprob*=GaussianModel(classmean[0][k], classvar[0][k], featureseval[j][k]);
			}
			if(spamprob==0)
			{
				spamprob=hamprob/10;
			}
			if(hamprob==0)
			{
				hamprob=spamprob/10;
			}
			if(Math.log(spamprob/hamprob)>threshold)
			{
				if(labelseval[j]==1)
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
				if(labelseval[j]==0)
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
		cc[0]=tp;
		cc[1]=fp;
		cc[2]=tn;
		cc[3]=fn;
		//System.out.println("Accuracy: "+acc*100+"  "+spam+"  "+ham+"  "+tp+"  "+fp+"  "+tn+"  "+fn);
		return acc;
	}
	private static double[][] Gaussian(double[][] cc,double[] acc, boolean mode){
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		double[] val={0,1};
		Double[][] classmean=getMeanbyClass(featurestrain, labelstrain, trainlen, featlen, val);
		Double[][] classvar=getVarbyClass(featurestrain, labelstrain, trainlen, featlen, val, classmean);
		int spamcount=0;
		 for (int j = 0; j < trainlen; j++) {
			 spamcount+=labelstrain[j];
		 }
		 double spam=((double) spamcount)/((double) trainlen),ham=1-spam;
		 //evaluate
		 if(!mode)
		 {
			 acc[0]=evaluateGaussian(featurestrain, labelstrain, classmean, classvar, cc[0], spam, ham, trainlen,0.0);
			 acc[1]=evaluateGaussian(featurestest, labelstest, classmean, classvar, cc[1], spam, ham, testlen,0.0);
		 }
		 else
		 {
			 return ROCGaussian(featurestest, labelstest, classmean, classvar, spam, ham, testlen);
		 }
		 return null;
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
			System.out.println(i);
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
				System.out.println(a.get(i, j));
				//System.out.print(" ");
			}
			System.out.println();
		}
	}
	
	public static void main(String[] args) {
		readPolluted();
		noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		globalmean=getMean(featurestrain, trainlen, featlen);
		double[][] classconfusion = new double[2][4];
		double accuracy[]= new double[2];
		//Bernoulli(classconfusion, accuracy,false);
		Gaussian(classconfusion, accuracy, false);
		System.out.println("Training Accuracy:"+accuracy[0]*100+"\nTesting Accuracy:"+accuracy[1]*100);
		
		
		
        
	}

}
