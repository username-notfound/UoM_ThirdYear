/** 

 * Copyright (C) University of Manchester - All Rights Reserved
 * Unauthorised copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Kristian Epps <kristian@xepps.com>, August 28, 2013
 * 
 * RDF Inverted Index
 * 
 * This Map Reduce program should read in a set of RDF/XML documents and output
 * the data in the form:
 * 
 * {predicate, object]}, [subject1, subject2, ...] 
 * 
 * @author Kristian Epps
 * 
 */
package uk.ac.man.cs.comp38211.exercise;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.StringTokenizer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.hp.hpl.jena.rdf.model.Model;
import com.hp.hpl.jena.rdf.model.ModelFactory;
import com.hp.hpl.jena.rdf.model.RDFNode;
import com.hp.hpl.jena.rdf.model.Resource;
import com.hp.hpl.jena.rdf.model.Literal;
import com.hp.hpl.jena.rdf.model.Statement;
import com.hp.hpl.jena.rdf.model.StmtIterator;
import com.hp.hpl.jena.rdf.model.Property;

import uk.ac.man.cs.comp38211.io.array.ArrayListWritable;
import uk.ac.man.cs.comp38211.io.pair.PairOfStrings;
import uk.ac.man.cs.comp38211.util.XParser;
import uk.ac.man.cs.comp38211.ir.StopAnalyser;

public class RDFInvertedIndex extends Configured implements Tool
{
  private static final Logger LOG = Logger
      .getLogger(RDFInvertedIndex.class);

  public static class Map extends 
      Mapper<LongWritable, Text, PairOfStrings, Text>
  {        

    protected Text document = new Text();
    protected PairOfStrings predobj = new PairOfStrings();
    protected Text subj = new Text();

    private void indexing(Context c, Resource s, Property p, RDFNode o) throws IOException, InterruptedException {
          String pre = p.toString();
          String obj = o.toString();
          String sub = s.toString();
          predobj.set(pre, obj);
          subj.set(sub);
          c.write(predobj, subj);
        }
 
    private String caseFolding(String token) {
    	// Condition1: a word with the first letter in upper case
    	// it is more likely that it is the first word of a sentence, which means we can turn it to lower case
    	// if the capital letter is in the middle, it may because of mistyping, we can still turn it to lower case
    	// Condition2: all the letter in this word is in upper case
    	// we probably should keep the form, such as BBC, DVD
    	int numOfCapitalLetter = 0;
    	char[] c = token.toCharArray();
    	for (int i = 0; i < token.length(); i++) {
  
    		if (Character.isUpperCase(c[i])) {numOfCapitalLetter++; }
    		if (numOfCapitalLetter == token.length()) 
    			return token;
    	}
    	return token.toLowerCase();
    } // case folding
    
    // The StopAnalyser class helps remove stop words
    //@SuppressWarnings("unused")
    //private StopAnalyser stopAnalyser = new StopAnalyser();
    
    public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException
    {            
      // This statement ensures we read a full rdf/xml document in
      // before we try to do anything else
      if(!value.toString().contains("</rdf:RDF>"))
      {    
        document.set(document.toString() + value.toString());
        return;
      }
      
      // We have to convert the text to a UTF-8 string. This must be
      // enforced or errors will be thrown. 
      String contents = document.toString() + value.toString();
      contents = new String(contents.getBytes(), "UTF-8");
      
      // The string must be cast to an inputstream for use with jena
      InputStream fullDocument = IOUtils.toInputStream(contents);
      document = new Text();
      
      // Create a model
      Model model = ModelFactory.createDefaultModel();
      
      try
      {
        model.read(fullDocument, null);
      
        StmtIterator iter = model.listStatements();
      
        // Iterate over triples
        while(iter.hasNext())
        {
          Statement stmt      = iter.nextStatement();
          Resource  subject   = stmt.getSubject();
          Property  predicate = stmt.getPredicate();
          RDFNode   object    = stmt.getObject();
          
          if (object.isLiteral()) { // Check if the RDFNode is resource or literal
            Literal li = object.asLiteral();
            
            if (li.getDatatype() == null)
            {  
              String le = li.getLexicalForm(); // Get the lexical form
              StringTokenizer stringTokenizer = new StringTokenizer(le);
             
              while (stringTokenizer.hasMoreTokens())
              { 
            	  // Remove all non-alphabetic characters
                String token = stringTokenizer.nextToken();
                token = token.replaceAll("[^a-zA-Z]", "");
                //caseFolding(token);
                
                // Making the token type as literal
                Literal liOriginalToken = model.createLiteral(token);
                Literal liLowerCaseToken = model.createLiteral(token.toLowerCase());


                // The only one-letter words in English are a and I, which is not a good token
        		    // So there I check the length of token to filter white space and a/I
                if (token.length() > 1) 
                {
                  indexing(context, subject, predicate, liOriginalToken);
                  indexing(context, subject, predicate, liLowerCaseToken);
                } // if
              } // while 
            } else {
            	// Deal with the number (e.g. integer, date)
              indexing(context, subject, predicate, li);
            } // else
          } // if 
        } // while
      } // try
      catch(Exception e)
      {
        LOG.error(e);
      } // catch
    } // map
  } // Map

  public static class Reduce extends Reducer<PairOfStrings, Text, PairOfStrings, ArrayListWritable<Text>>
  {
     
    // This reducer turns an iterable into an ArrayListWritable, sorts it
    // and outputs it
    public void reduce(
        PairOfStrings key,
        Iterable<Text> values,
        Context context) throws IOException, InterruptedException
    {
      ArrayListWritable<Text> postings = new ArrayListWritable<Text>();
      
      Iterator<Text> iter = values.iterator();
      
      while(iter.hasNext()) {
        Text copy = new Text(iter.next());
        postings.add(copy);
      }
      
      Collections.sort(postings);
      
      context.write(key, postings);
    }
  }

  public RDFInvertedIndex()
  {
  }

  private static final String INPUT = "input";
  private static final String OUTPUT = "output";
  private static final String NUM_REDUCERS = "numReducers";

  @SuppressWarnings({ "static-access" })
  public int run(String[] args) throws Exception
  {        
    Options options = new Options();

    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("input path").create(INPUT));
    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("output path").create(OUTPUT));
    options.addOption(OptionBuilder.withArgName("num").hasArg()
        .withDescription("number of reducers").create(NUM_REDUCERS));

    CommandLine cmdline = null;
    CommandLineParser parser = new XParser(true);

    try
    {
      cmdline = parser.parse(options, args);
    }
    catch (ParseException exp)
    {
      System.err.println("Error parsing command line: "
          + exp.getMessage());
      System.err.println(cmdline);
      return -1;
    }

    if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT))
    {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(this.getClass().getName(), options);
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }      
    
    String inputPath = cmdline.getOptionValue(INPUT);
    String outputPath = cmdline.getOptionValue(OUTPUT);

    Job RDFIndex = new Job(new Configuration());

    RDFIndex.setJobName("Inverted Index 1");
    RDFIndex.setJarByClass(RDFInvertedIndex.class);        
    RDFIndex.setMapperClass(Map.class);
    RDFIndex.setReducerClass(Reduce.class);
    RDFIndex.setMapOutputKeyClass(PairOfStrings.class);
    RDFIndex.setMapOutputValueClass(Text.class);
    RDFIndex.setOutputKeyClass(PairOfStrings.class);
    RDFIndex.setOutputValueClass(ArrayListWritable.class);
    FileInputFormat.setInputPaths(RDFIndex, new Path(inputPath));
    FileOutputFormat.setOutputPath(RDFIndex, new Path(outputPath));

    long startTime = System.currentTimeMillis();
     
    RDFIndex.waitForCompletion(true);
    if(RDFIndex.isSuccessful())
      LOG.info("Job successful!");
    else
      LOG.info("Job failed.");
    
    LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime)
        / 1000.0 + " seconds");

    return 0;
  }

  public static void main(String[] args) throws Exception
  {
    ToolRunner.run(new RDFInvertedIndex(), args);
  }
}
