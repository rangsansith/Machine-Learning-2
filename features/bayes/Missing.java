package hw5.features.bayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

public class Missing {

	static int len=4601;
	static int featlen=57;
	static int classno=2;
	static int trainlen,testlen;
	static double[][] featurestrain = new double[1][featlen];
	static double [] labelstrain = new double[1];
	static double [][] featurestest= new double[1][featlen];
	static double [] labelstest= new double[1];
	static Double[] globalmean;
	private static void readMissing()
	{
		int one=0,zero=0;
		trainlen=3681;
		testlen=len-trainlen;
		featurestrain = new double[trainlen][featlen];
		labelstrain=new double[trainlen];
		featurestest = new double[testlen][featlen];
		labelstest  = new double[testlen];
		try {
			FileReader featureread = new FileReader("data/spam_missing/20_percent_missing_train.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					if(!feats[i].contentEquals("nan"))
						featurestrain[ind][i]=Double.parseDouble(feats[i]);
					else
						featurestrain[ind][i]=Double.NaN;	

				}
				labelstrain[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labelstrain[ind]==0.0)
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
		one=0;zero=0;
		try {
			FileReader featureread = new FileReader("data/spam_missing/20_percent_missing_test.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(",");
				for(int i=0;i<feats.length-1;i++)
				{
					if(!feats[i].contentEquals("nan"))
						featurestest[ind][i]=Double.parseDouble(feats[i]);
					else
						featurestest[ind][i]=Double.NaN;		
				}
				labelstest[ind]=Double.parseDouble(feats[feats.length-1]);
				if(labelstest[ind]==0.0)
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
	public static void main(String[] args) {
		readMissing();
		//noramalizeboth(featurestrain, featurestest, trainlen, testlen, featlen);
		globalmean=getMean(featurestrain, trainlen, featlen);
		double[][] classconfusion = new double[2][4];
		double accuracy[]= new double[2];
		Bernoulli(classconfusion, accuracy,false);
		System.out.println("Training Accuracy:"+accuracy[0]*100+"\nTesting Accuracy:"+accuracy[1]*100);
	}


}
