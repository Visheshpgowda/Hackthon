import mongoose from 'mongoose';

const marksSchema = new mongoose.Schema({
  studentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Student',
    required: true
  },
  roll_no: {
    type: Number,
    required: true
  },
  subjectName: {
    type: String, 
    required: true
  },
  examType: {
    type: String,
    enum: ['Unit Test', 'Mid Term', 'Final Term'],
    required: true
  },
  marks: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  class: {
    type: String,
    required: true
  },
  teacherId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Teacher',
    required: true
  },
  date: {
    type: Date,
    required: true
  }
}, {
  timestamps: true
});
export const Marks = mongoose.model('Marks', marksSchema);