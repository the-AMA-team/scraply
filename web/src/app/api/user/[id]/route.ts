import { NextRequest, NextResponse } from "next/server";
import prisma from "../../../../../utils/connect";

export const GET = async (req: NextRequest, {params} : any) => {
    try {
      const { id } = await params;
      const user = await prisma.user.findUnique({
        where: {
          id: id
        }
      });
  
      // if (!user) {
      //   console.log("User not found");
      //   return null;
      // }
  
      // console.log("User found:", user);
      return NextResponse.json({ user: user}, {status: 200});
    } catch (error) {
      console.error("Error retrieving user:", error);
      return NextResponse.json({ error }, {status: 200});

      throw error;
    }
  }