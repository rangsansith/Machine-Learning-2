package hw5.features.haar;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import Jama.Matrix;
import edu.uci.lasso.LassoFit;
import edu.uci.lasso.LassoFitGenerator;

public class Helper {

	static int len=70000;
	static int featlen=200;
	static int classno=10;
	static int trainlen=60000,testlen=10000;
	static int codesize=511;
	static int stop=100;
	static int []classcount=new int[classno];
	static double [] labelstrain = new double[trainlen];
	static double [] labelstest= new double[testlen];
	static double [] codelabelstrain = new double[trainlen];
	static double [] codelabelstest= new double[testlen];
	static double [] predictiontest;
	static double [] predictiontrain;
	static double[][] totalpredictiontest = new double[codesize][testlen];
	static double[][] totalpredictiontrain = new double[codesize][trainlen];
	static double dataweights[];
	static double[][] featurestrain = new double[trainlen][featlen];
	static double[][] featurestest = new double[testlen][featlen];
	static ArrayList<double[]> codelabels = new ArrayList<double[]>();
	static ArrayList<ArrayList<Double>> stumps=new ArrayList<ArrayList<Double>>();
	static ArrayList<Double> alphat;
	static ArrayList<Integer> stumpt;
	static ArrayList<Double> threstt;
	static ArrayList<Double> errort;
	static ArrayList<Double> auct;
	static ArrayList<Double> trainerrort;
	static ArrayList<Double> testerrort;
	static int runs;
	static int totalstump;
	private static void readData()
	{
		try {
			FileReader featureread = new FileReader("data/digitsdataset/trainfeat200.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestrain[ind][i]=Double.parseDouble(feats[i]);	
				}
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/digitsdataset/trainlabels.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				labelstrain[ind]=Double.parseDouble(sCurrentLine);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/digitsdataset/testfeat200.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			String[] feats;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				feats=sCurrentLine.split(" ");
				for(int i=0;i<feats.length;i++)
				{
					featurestest[ind][i]=Double.parseDouble(feats[i]);	
				}
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
		try {
			FileReader featureread = new FileReader("data/digitsdataset/testlabels.txt");
			BufferedReader featurereadbr = new BufferedReader(featureread);
			String sCurrentLine;
			int ind=0;
			while ((sCurrentLine = featurereadbr.readLine()) != null) {
				labelstest[ind]=Double.parseDouble(sCurrentLine);
				ind++;
			}
			featurereadbr.close();
			featureread.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	private static void writeforlib(){

		try {
			FileWriter fw = new FileWriter("data/digitsdataset/trainlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < trainlen; i++) {
				bw.write(((int)labelstrain[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestrain[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		try {
			FileWriter fw = new FileWriter("data/digitsdataset/testlib.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < testlen; i++) {
				bw.write(((int)labelstest[i])+"");
				for (int j = 0; j < featlen; j++) {
					bw.write(" "+(j+1)+":"+featurestest[i][j]);
				}
				bw.newLine();
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}
	private static void classcnt(){
		for(int i=0;i<classno;i++)
			classcount[i]=0;
		for (int i = 0; i < trainlen; i++) {
			classcount[(int) labelstrain[i]]++;
		}
		for (int i = 0; i < classno; i++) {
			System.out.println(i+"  "+classcount[i]);
		}
	}
	private static void sample(int limit){
		int newcnt[]=new int[10];
		for (int i = 0; i < newcnt.length; i++) {
			newcnt[i]=0;
		}
		double[][] newtrainfeat = new double[limit*classno][featlen];
		double[] newtrainlab = new double[limit*classno];
		int ind=0;
		for (int i = 0; i < trainlen; i++) {
			if(newcnt[(int) labelstrain[i]]<limit)
			{
				newcnt[(int) labelstrain[i]]++;
				newtrainfeat[ind]=featurestrain[i];
				newtrainlab[ind]=labelstrain[i];
				ind++;
			}
		}
		featurestrain=newtrainfeat;
		labelstrain=newtrainlab;
		trainlen=limit*classno;
		
	}
	private static void writeforecoc(){
		try {
			FileWriter fw = new FileWriter("data/digitsdataset/trainecoc.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < trainlen; i++) {
				for (int j = 0; j < featlen; j++) {
					bw.write(featurestrain[i][j]+" ");
				}
				bw.write(((int)labelstrain[i])+"\n");
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
		try {
			FileWriter fw = new FileWriter("data/digitsdataset/testecoc.txt");
			BufferedWriter bw = new BufferedWriter(fw);
			for (int i = 0; i < testlen; i++) {
				for (int j = 0; j < featlen; j++) {
					bw.write(featurestest[i][j]+" ");
				}
				bw.write(((int)labelstest[i])+"\n");
			}
			bw.close();
			fw.close();

		}catch (Exception e){
			System.out.println(e.getMessage());	
		}
	}
	public static void main(String[] args) {

		readData();
		classcnt();
		sample(500);
		classcnt();
		writeforlib();
		writeforecoc();
	}
}
