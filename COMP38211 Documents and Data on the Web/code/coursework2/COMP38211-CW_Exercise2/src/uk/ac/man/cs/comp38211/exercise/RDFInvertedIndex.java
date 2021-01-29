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
package uk.ac.man.cs.comp38120.exercise;

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

import uk.ac.man.cs.comp38120.io.array.ArrayListWritable;
import uk.ac.man.cs.comp38120.io.pair.PairOfStrings;
import uk.ac.man.cs.comp38120.util.XParser;

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

    private void index(Context c, Resource  s, Property p,
            RDFNode o, String type) {
          String flag = type + ":";    
          predobj.set(
              p.toString(),
              flag + o.toString()
            );
          subj.set(s.toString());
          try {
			c.write(predobj, subj);
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
		}
        }
 
    public void map(LongWritable key, Text value, Context context)
        throws IOException, InterruptedException
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
          
          if (object instanceof Literal) 
          {
            Literal l = object.asLiteral();

            // Checking the literal type
            if (l.getDatatype() == null)
            {  
              // Declaring and retrieving the lexical form
              String lexicalFormLiteral = l.getLexicalForm();
              
              // Declaring and retrieving the language
              String languageLiteral = l.getLanguage();
              
              // Declaring the tokenizer that will iterate through them
              StringTokenizer interating = new StringTokenizer(lexicalFormLiteral);
              
              while (interating.hasMoreTokens())
              { 
            	// Declaring and filtering the tokens so that only strings will remain
                String theToken = interating.nextToken().replaceAll("[^a-zA-Z0-9]", "");
                
                // Declaring and creating model for case insensitive tokens
                Literal insensitiveToken = model.createLiteral(theToken.toLowerCase(), languageLiteral);
                
                // Declaring and creating model for case sensitive tokens
                Literal sensitiveToken = model.createLiteral(theToken, languageLiteral);

                // Checking the token for the plain literal
                if (theToken.length() > 1 && theToken != null && !theToken.isEmpty()) 
                {
                  // Indexing the plain literals that are case insensitive
                  index(context, subject, predicate, insensitiveToken, "literal-plain-case-insensitive");
                  
                  // Indexing the plain literals that are case sensitive
                  index(context, subject, predicate, sensitiveToken, "literal-plain-case-sensitive");
                } // if
              } // while 
            } // if 
            else
            {
              // Indexing the typed literals + flagging them
              index(context, subject, predicate, l, "literal-typed");
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
